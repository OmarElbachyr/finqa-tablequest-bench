#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Script

Evaluates answer correctness for FinanceBench and TableQuest datasets using LLM-as-a-judge.
Uses qwen2.5:14b as the judge model to assess generated answers against ground truth.
"""

import json
import os
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import ollama


def load_generated_answers(file_path: str) -> List[Dict[str, Any]]:
    """Load generated answers from JSON file."""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        return json.load(f)


def judge_answer_with_ollama(judge_model: str, question: str, ground_truth: str, generated_answer: str, dataset: str) -> Tuple[str, float, str]:
    """
    Use LLM-as-a-judge to evaluate answer correctness.
    Returns: (judgment, confidence_score, reasoning)
    """
    
    # Simplified unified prompt for both datasets
    system_prompt = """You are an expert judge tasked with evaluating answer correctness. Determine if the generated answer is correct compared to the ground truth answer."""

    prompt_template = """
QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth}

GENERATED ANSWER: {generated_answer}

Evaluate if the GENERATED ANSWER is correct compared to the GROUND TRUTH ANSWER.

Respond in the following format:
JUDGMENT: [CORRECT/INCORRECT]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation of your decision]
"""
    
    formatted_prompt = prompt_template.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer
    )
    
    try:
        response = ollama.chat(
            model=judge_model,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': formatted_prompt
                }
            ],
            options={
                'temperature': 0.1,
                'num_predict': 512
            }
        )
        
        response_text = response['message']['content']
        
        # Parse the response
        judgment = "INCORRECT"  # Default
        confidence = 0.0
        reasoning = "Failed to parse response"
        
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('JUDGMENT:'):
                judgment_text = line.replace('JUDGMENT:', '').strip()
                if 'CORRECT' in judgment_text.upper() and 'INCORRECT' not in judgment_text.upper():
                    judgment = "CORRECT"
                else:
                    judgment = "INCORRECT"
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                except:
                    confidence = 0.0
            elif line.startswith('REASONING:'):
                reasoning = line.replace('REASONING:', '').strip()
        
        return judgment, confidence, reasoning
        
    except Exception as e:
        if 'timeout' in str(e).lower():
            return "ERROR", 0.0, "timeout"
        return "ERROR", 0.0, f"Error: {str(e)}"


def evaluate_generated_answers(answers: List[Dict[str, Any]], judge_model: str, dataset: str) -> List[Dict[str, Any]]:
    """Evaluate a list of generated answers using LLM-as-a-judge."""
    evaluated_answers = []
    
    for answer_data in tqdm(answers, desc=f"  Judging with {judge_model}"):
        question = answer_data["question"]
        ground_truth = answer_data["ground_truth_answer"]
        generated_answer = answer_data["generated_answer"]
        
        start_time = datetime.datetime.now()
        judgment, confidence, reasoning = judge_answer_with_ollama(
            judge_model, question, ground_truth, generated_answer, dataset
        )
        end_time = datetime.datetime.now()
        judgment_time = (end_time - start_time).total_seconds()
        
        # Copy original data and add judgment results
        evaluated_data = answer_data.copy()
        evaluated_data.update({
            "judgment": judgment,
            "judgment_confidence": confidence,
            "judgment_reasoning": reasoning,
            "judgment_time": judgment_time,
            "judge_model": judge_model
        })
        
        evaluated_answers.append(evaluated_data)
    
    return evaluated_answers


def calculate_accuracy_metrics(evaluated_answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy metrics from evaluated answers."""
    if not evaluated_answers:
        return {}
    
    total_questions = len(evaluated_answers)
    correct_answers = sum(1 for ans in evaluated_answers if ans.get("judgment") == "CORRECT")
    incorrect_answers = sum(1 for ans in evaluated_answers if ans.get("judgment") == "INCORRECT")
    error_answers = sum(1 for ans in evaluated_answers if ans.get("judgment") == "ERROR")
    
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    
    # Calculate average confidence
    confidences = [ans.get("judgment_confidence", 0.0) for ans in evaluated_answers if ans.get("judgment") != "ERROR"]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Calculate average judgment time
    judgment_times = [ans.get("judgment_time", 0.0) for ans in evaluated_answers]
    avg_judgment_time = sum(judgment_times) / len(judgment_times) if judgment_times else 0.0
    
    return {
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "incorrect_answers": incorrect_answers,
        "error_answers": error_answers,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "avg_judgment_time": avg_judgment_time
    }


def save_evaluation_results(evaluated_answers: List[Dict[str, Any]], metrics: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {
        "evaluation_metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_evaluated": len(evaluated_answers),
            "metrics": metrics
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved evaluation metadata to {output_path}")


def save_accuracy_summary(metrics: Dict[str, Any], combination_info: Dict[str, Any], output_path: str):
    """Save simplified accuracy summary for easy plotting/analysis."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    summary = {**combination_info, **metrics}
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(output_path)
    
    with open(output_path, 'a') as f:
        if not file_exists:
            # Write header
            headers = ["dataset", "parser", "chunker", "overlap", "retriever", "model", "top_k", 
                      "total_questions", "correct_answers", "accuracy", "avg_confidence", "avg_judgment_time"]
            f.write('\t'.join(headers) + '\n')
        
        # Write data row
        values = [
            str(summary.get("dataset", "")),
            str(summary.get("parser", "")),
            str(summary.get("chunker", "")),
            str(summary.get("overlap", "")),
            str(summary.get("retriever", "")),
            str(summary.get("model", "")),
            str(summary.get("top_k", "")),
            str(summary.get("total_questions", 0)),
            str(summary.get("correct_answers", 0)),
            f"{summary.get('accuracy', 0.0):.4f}",
            f"{summary.get('avg_confidence', 0.0):.4f}",
            f"{summary.get('avg_judgment_time', 0.0):.2f}"
        ]
        f.write('\t'.join(values) + '\n')


def check_paths(base_dir: Path, datasets: List[str]) -> bool:
    """Check if required paths exist."""
    generated_answers_base = base_dir / "data" / "generated_answers"
    
    if not generated_answers_base.exists():
        print(f"❌ Generated answers directory not found: {generated_answers_base}")
        return False
    
    # Check for dataset-specific paths
    for dataset in datasets:
        dataset_answers_dir = generated_answers_base / dataset
        if not dataset_answers_dir.exists():
            print(f"❌ Generated answers dataset directory not found: {dataset_answers_dir}")
            return False
        
        # Check for answer files in dataset
        overlap_dirs = list(dataset_answers_dir.glob("overlap_*"))
        if not overlap_dirs:
            print(f"❌ No overlap directories found in {dataset_answers_dir}")
            return False
        
        found_files = False
        for overlap_dir in overlap_dirs:
            parser_chunker_dirs = [d for d in overlap_dir.iterdir() if d.is_dir()]
            for pc_dir in parser_chunker_dirs:
                retriever_dirs = [d for d in pc_dir.iterdir() if d.is_dir()]
                for ret_dir in retriever_dirs:
                    answer_files = list(ret_dir.glob("*_answers.json"))
                    if answer_files:
                        print(f"✅ Found answer files in {dataset}/{pc_dir.name}/{ret_dir.name}")
                        found_files = True
        
        if not found_files:
            print(f"❌ No answer files found for dataset: {dataset}")
            return False
    
    return True


def check_existing_result(summary_file: str, combination_info: Dict[str, Any]) -> bool:
    """Check if a result already exists in the summary file for this combination."""
    if not os.path.exists(summary_file):
        return False
    
    try:
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            
        if len(lines) <= 1:  # Only header or empty file
            return False
            
        # Check each line for matching combination
        for line in lines[1:]:  # Skip header
            fields = line.strip().split('\t')
            if len(fields) >= 7:  # Ensure we have enough fields
                if (fields[0] == combination_info["dataset"] and
                    fields[1] == combination_info["parser"] and
                    fields[2] == combination_info["chunker"] and
                    fields[3] == str(combination_info["overlap"]) and
                    fields[4] == combination_info["retriever"] and
                    fields[5] == combination_info["model"] and
                    fields[6] == str(combination_info["top_k"])):
                    return True
        return False
    except Exception:
        return False


def process_combination(parser: str, chunker: str, overlap: int, retriever: str, model: str, top_k: int,
                       base_dir: Path, judge_model: str, dataset: str = "financebench") -> None:
    """Process a single combination of parser, chunker, overlap, retriever, and model."""
    print(f"Processing: {parser}_{chunker} | overlap_{overlap} | {retriever} | {model} | top_k={top_k} | dataset={dataset}")
    
    # Construct paths
    generated_answers_base = base_dir / "data" / "generated_answers" / dataset
    evaluation_output_base = base_dir / "evaluation" / "gen_accuracy" / dataset
    
    # Input file (generated answers)
    model_filename = model.replace(':', '_')
    input_file = generated_answers_base / f"overlap_{overlap}" / f"{parser}_{chunker}" / retriever / f"{model_filename}_top{top_k}_answers.json"
    
    # Output files
    output_dir = evaluation_output_base / f"overlap_{overlap}" / f"{parser}_{chunker}" / retriever
    detailed_output_file = output_dir / f"{model_filename}_top{top_k}_{judge_model.replace(':', '_')}_evaluation.json"
    summary_output_file = base_dir / "evaluation" / "gen_accuracy" / f"{dataset}_accuracy_summary.tsv"
    
    # Check if results already exist (both detailed file and summary entry)
    combination_info = {
        "dataset": dataset,
        "parser": parser,
        "chunker": chunker,
        "overlap": overlap,
        "retriever": retriever,
        "model": model,
        "top_k": top_k
    }
    
    if detailed_output_file.exists() or check_existing_result(str(summary_output_file), combination_info):
        print(f"  Skipping - results already exist for this combination")
        return
    
    # Load generated answers
    generated_answers = load_generated_answers(str(input_file))
    if not generated_answers:
        print(f"  Skipping - no generated answers found in {input_file}")
        return
    
    print(f"  Loaded {len(generated_answers)} generated answers")
    
    # Evaluate answers using LLM-as-a-judge
    evaluated_answers = evaluate_generated_answers(generated_answers, judge_model, dataset)
    
    # Calculate metrics
    metrics = calculate_accuracy_metrics(evaluated_answers)
    
    # Save detailed results
    save_evaluation_results(evaluated_answers, metrics, str(detailed_output_file))
    
    # Save summary for plotting/analysis
    save_accuracy_summary(metrics, combination_info, str(summary_output_file))
    
    print(f"  ✅ Completed: {metrics['correct_answers']}/{metrics['total_questions']} correct (accuracy: {metrics['accuracy']:.3f})")


def main(parsers: List[str], chunkers: List[str], overlaps: List[int], 
         retrievers: List[str], models: List[str], top_ks: List[int],
         base_dir: Path, judge_model: str, datasets: List[str]):
    """Main processing function."""
    for dataset in datasets:
        print(f"\n=== Evaluating dataset: {dataset} ===")
        
        # Calculate and process all combinations
        total_combinations = len(overlaps) * len(retrievers) * len(models) * len(parsers) * len(chunkers) * len(top_ks)
        current_combination = 0
        
        for top_k in top_ks:
            for model in models:
                for overlap in overlaps:
                    for retriever in retrievers:
                        for parser in parsers:
                            for chunker in chunkers:
                                current_combination += 1
                                combination_start = datetime.datetime.now()
                                print(f"\n[{combination_start}] Starting combination ({current_combination}/{total_combinations})")
                                process_combination(parser, chunker, overlap, retriever, model, top_k, base_dir, judge_model, dataset)
                                combination_end = datetime.datetime.now()
                                duration = combination_end - combination_start
                                print(f"  Duration: {duration}")


if __name__ == "__main__":
    script_start = datetime.datetime.now()
    print(f"🚀 Script started at: {script_start}")
    
    # =============================================================================
    # CONFIGURATION - Edit these variables as needed
    # =============================================================================
    
    # Base directory
    BASE_DIR = Path(".")
    
    # Judge model
    JUDGE_MODEL = "qwen2.5:14b"
    
    # Configuration lists - edit these to change what gets processed
    DATASETS = ["financebench", "tablequest"]
    PARSERS = ['pdfminer']
    CHUNKERS = ['sentence']
    OVERLAPS = [128]
    RETRIEVERS = ["SpladeRetriever"]  # Add: "ColBERTRetriever", "SentenceTransformerRetriever", "SpladeRetriever"
    MODELS = ["gemma3:4b", "phi3:mini", "llama3.2:3b", "deepseek-r1:1.5b"]
    TOP_KS = [1, 3]  # Top-K values to evaluate
    
    # =============================================================================
    # END CONFIGURATION
    # =============================================================================
    
    print(f"Configuration:")
    print(f"  Judge Model: {JUDGE_MODEL}")
    print(f"  Parsers: {PARSERS}")
    print(f"  Chunkers: {CHUNKERS}")
    print(f"  Overlaps: {OVERLAPS}")
    print(f"  Retrievers: {RETRIEVERS}")
    print(f"  Models: {MODELS}")
    print(f"  Top-K values: {TOP_KS}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Total combinations: {len(OVERLAPS) * len(RETRIEVERS) * len(MODELS) * len(PARSERS) * len(CHUNKERS) * len(TOP_KS) * len(DATASETS)}")
    
    # Validate setup
    if not check_paths(BASE_DIR, DATASETS):
        print("❌ Setup validation failed. Please fix the issues before running.")
        exit(1)
    
    print("✅ Setup validation passed. Starting evaluation...")
    
    # Run main processing
    main(PARSERS, CHUNKERS, OVERLAPS, RETRIEVERS, MODELS, TOP_KS, BASE_DIR, JUDGE_MODEL, DATASETS)
    
    script_end = datetime.datetime.now()
    overall_duration = script_end - script_start
    print(f"\n🎉 Script finished at: {script_end}")
    print(f"Total execution time: {overall_duration}")
