#!/usr/bin/env python3
"""
Ollama Answer Generation Script

Generates answers for document QA pairs using Ollama models with direct API calls.
Uses configurable lists for parsers, chunkers, overlaps, retrievers, and models.
"""

import json
import os
import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import ollama

from haystack import Pipeline
from haystack.components.builders import PromptBuilder

from requests.exceptions import ReadTimeout


def load_qa_pairs(csv_path: str) -> List[Dict[str, Any]]:
    """Load QA pairs from JSON file."""
    with open(csv_path, 'r') as f:
        return json.load(f)


def load_retrieved_pages(retrieval_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load retrieved pages from a retrieval run file."""
    if not os.path.exists(retrieval_file_path):
        return {}
    
    with open(retrieval_file_path, 'r') as f:
        return json.load(f)


def load_parsed_page_content(page_name: str, parser: str, base_dir: Path, dataset: str) -> str:
    """Load content from parsed pages directory for a specific page name (e.g., BESTBUY_2024Q2_10Q_20)."""
    parsed_pages_dir = base_dir / "data" / "parsed_pages" / dataset / parser
    
    # Try the original format first (financebench format)
    page_file = parsed_pages_dir / f"{page_name}_{parser}.json"
    
    # If not found, try the tablequest format with 'p' prefix
    if not page_file.exists() and dataset == "tablequest":
        # For tablequest, convert "DOCUMENT_PAGE" to "DOCUMENT_pPAGE"
        parts = page_name.rsplit('_', 1)  # Split on last underscore
        if len(parts) == 2:
            doc_part, page_num = parts
            page_file = parsed_pages_dir / f"{doc_part}_p{page_num}_{parser}.json"
    
    if not page_file.exists():
        return ""
    
    try:
        with open(page_file, 'r') as f:
            data = json.load(f)
        
        # Extract text content from the parsed page
        if isinstance(data, list) and len(data) > 0:
            return data[0].get('text', '')
        elif isinstance(data, dict):
            return data.get('text', '')
        
        return ""
    except Exception:
        return ""


def build_context_from_retrieved_pages(retrieved_pages: Dict[str, float], parser: str, base_dir: Path, dataset: str, top_k: int = 5) -> str:
    """Build context string from top-k retrieved pages by loading content from parsed pages."""
    if not retrieved_pages:
        return ""
    
    # Get top-k page names (keys) sorted by score (highest first)
    top_page_names = list(retrieved_pages.keys())[:top_k]
    
    context_parts = []
    context_count = 0
    
    for page_name in top_page_names:
        content = load_parsed_page_content(page_name, parser, base_dir, dataset)
        if content.strip():
            context_count += 1
            context_parts.append(f"[CONTEXT {context_count}] - Source: {page_name}")
            context_parts.append(content.strip())
            context_parts.append("")  # Add blank line between contexts
    
    if context_count == 0:
        return ""
    
    # Add a header showing total number of context documents
    header = f"Total context documents retrieved: {context_count}\n{'='*60}\n\n"
    return header + "\n".join(context_parts)


def generate_answer_with_ollama(model: str, prompt: str, system_prompt: str = None) -> str:
    """Generate answer using direct Ollama API."""
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt or 'You are a financial expert specializing in corporate financial reports and filings.'
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.1,
                'num_predict': 256  # Limit output to 256 tokens for faster generation
            }
        )
        return response['message']['content']
    except Exception as e:
        if 'timeout' in str(e).lower():
            return "timeout"
        return f"Error: {str(e)}"


def create_generation_pipeline(model: str) -> Pipeline:
    """Create a Haystack pipeline for prompt building only."""
    template = """
Your task is to answer the QUESTION below using ONLY the information provided in the CONTEXT sections that follow. 

INSTRUCTIONS:
- Base your answer EXCLUSIVELY on the provided context documents
- Do not use any external knowledge or assumptions
- If the context does not contain sufficient information to answer the question, respond with "No answer"
- Be precise and cite specific information from the context when possible
- If information is found across multiple context sections, synthesize it appropriately

QUESTION: {{ query }}

CONTEXT DOCUMENTS:
{{ context }}

ANSWER:"""

    prompt_builder = PromptBuilder(template=template)
    
    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    
    return pipe


def generate_answer_with_pipeline(pipeline: Pipeline, question: str, context: str, model: str) -> str:
    """Generate answer using Haystack pipeline for prompt building and direct Ollama API."""
    if not context.strip():
        return "No retrieved context available"
    
    # Build the prompt using Haystack
    prompt_result = pipeline.get_component("prompt_builder").run(
        context=context,
        query=question
    )
    actual_prompt = prompt_result["prompt"]
    
    # Use direct Ollama API to generate the answer
    system_prompt = 'You are a financial expert specializing in corporate financial reports and filings.'
    generated_answer = generate_answer_with_ollama(model, actual_prompt, system_prompt)
    
    return generated_answer


def save_generated_answers(answers: List[Dict[str, Any]], output_path: str):
    """Save generated answers to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(answers, f, indent=2)
    
    print(f"Saved {len(answers)} answers to {output_path}")


def check_paths(base_dir: Path, datasets: List[str]) -> bool:
    """Check if required paths exist."""
    retrieved_pages_base = base_dir / "data" / "retrieved_pages"
    
    if not retrieved_pages_base.exists():
        print(f"❌ Retrieved pages directory not found: {retrieved_pages_base}")
        return False
    
    # Check for dataset-specific paths
    for dataset in datasets:
        if dataset == "financebench":
            csv_path = base_dir / "data" / "csv" / "document_qa_pairs.json"
        elif dataset == "tablequest":
            csv_path = base_dir / "data" / "csv" / "tq_document_qa_pairs.json"
        else:
            csv_path = base_dir / "data" / "csv" / dataset / "document_qa_pairs.json"
        
        if not csv_path.exists():
            print(f"❌ QA pairs file not found: {csv_path}")
            return False
        
        dataset_retrieved_pages = retrieved_pages_base / dataset
        if not dataset_retrieved_pages.exists():
            print(f"❌ Retrieved pages dataset directory not found: {dataset_retrieved_pages}")
            return False
        
        # Check for retriever result files in dataset
        overlap_dirs = list(dataset_retrieved_pages.glob("overlap_*"))
        if not overlap_dirs:
            print(f"❌ No overlap directories found in {dataset_retrieved_pages}")
            return False
        
        found_results = False
        for overlap_dir in overlap_dirs:
            parser_chunker_dirs = [d for d in overlap_dir.iterdir() if d.is_dir()]
            for pc_dir in parser_chunker_dirs:
                retriever_files = list(pc_dir.glob("*_run_sorted.json"))
                if retriever_files:
                    print(f"✅ Found retriever results in {dataset}/{pc_dir.name}")
                    found_results = True
        
        if not found_results:
            print(f"❌ No retriever result files found for dataset: {dataset}")
            return False
    
    return True

def process_combination(parser: str, chunker: str, overlap: int, retriever: str, model: str, 
                       base_dir: Path, all_questions: List[Dict[str, Any]], top_k: int = 5, dataset: str = "financebench") -> None:
    """Process a single combination of parser, chunker, overlap, retriever, and model."""
    print(f"Processing: {parser}_{chunker} | overlap_{overlap} | {retriever} | {model} | top_k={top_k} | dataset={dataset}")
    
    # Construct paths with dataset subfolder
    retrieved_pages_base = base_dir / "data" / "retrieved_pages" / dataset
    output_base = base_dir / "data" / "generated_answers" / dataset
    retrieval_file = retrieved_pages_base / f"overlap_{overlap}" / f"{parser}_{chunker}" / f"{retriever}_run_sorted.json"
    
    # Check if output file already exists
    output_dir = output_base / f"overlap_{overlap}" / f"{parser}_{chunker}" / retriever
    output_file = output_dir / f"{model.replace(':', '_')}_top{top_k}_answers.json"
    if output_file.exists():
        print(f"  Skipping - output file already exists: {output_file}")
        return
    
    # Load retrieved pages
    retrieved_data = load_retrieved_pages(str(retrieval_file))
    if not retrieved_data:
        print(f"  Skipping - no retrieved data found")
        return
    
    # Create pipeline (for prompt building only)
    pipeline = create_generation_pipeline(model)
    print(f"  ✅ Created pipeline for model: {model}")
    
    # Generate answers
    generated_answers = []
    for question_data in tqdm(all_questions, desc=f"  {model}"):
        question_id = question_data["question_id"]
        question = question_data["question"]
        ground_truth = question_data["answer"]
        doc_name = question_data["doc_name"]
        
        # Get retrieved pages for this question
        retrieved_pages = retrieved_data.get(question_id, {})
        
        start_gen_time = datetime.datetime.now()
        if not retrieved_pages:
            generated_answer = "No retrieved context available"
        else:
            # Build context from top-k retrieved pages using parsed page content
            context = build_context_from_retrieved_pages(retrieved_pages, parser, base_dir, dataset, top_k)
            generated_answer = generate_answer_with_pipeline(pipeline, question, context, model)
        end_gen_time = datetime.datetime.now()
        generation_time = (end_gen_time - start_gen_time).total_seconds()
        
        # Store result
        generated_answers.append({
            "question_id": question_id,
            "question": question,
            "ground_truth_answer": ground_truth,
            "generated_answer": generated_answer,
            "doc_name": doc_name,
            "retriever": retriever,
            "model": model,
            "parser": parser,
            "chunker": chunker,
            "overlap": overlap,
            "top_k": top_k,
            "num_retrieved_pages": len(retrieved_pages),
            "generation_time": generation_time
        })
    
    # Save results
    save_generated_answers(generated_answers, str(output_file))
    print(f"  ✅ Completed: {len(generated_answers)} answers generated")


def main(parsers: List[str], chunkers: List[str], overlaps: List[int], 
         retrievers: List[str], models: List[str], base_dir: Path, top_k: int, datasets: List[str]):
    """Main processing function."""
    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        # Load QA pairs
        if dataset == "financebench":
            csv_path = base_dir / "data" / "csv" / "document_qa_pairs.json"
        elif dataset == "tablequest":
            csv_path = base_dir / "data" / "csv" / "tq_document_qa_pairs.json"
        else:
            csv_path = base_dir / "data" / "csv" / dataset / "document_qa_pairs.json"
        
        print(f"Loading QA pairs from {csv_path}")
        qa_data = load_qa_pairs(str(csv_path))
        print(f"Loaded {len(qa_data)} documents with QA pairs")
        
        # Extract all questions
        all_questions = []
        question_counter = 1  # Start from 1 to match q1, q2, etc.
        for doc in qa_data:
            doc_name = doc["doc_name"]
            for qa_pair in doc["qa_pairs"]:
                question_id = f"q{question_counter}"  # Match the format in retrieved pages
                all_questions.append({
                    "question_id": question_id,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "doc_name": doc_name
                })
                question_counter += 1
        
        print(f"Total questions to process: {len(all_questions)}")
        
        # Calculate and process all combinations
        total_combinations = len(overlaps) * len(retrievers) * len(models) * len(parsers) * len(chunkers)
        current_combination = 0
        
        for model in models:
            for overlap in overlaps:
                for retriever in retrievers:
                    for parser in parsers:
                        for chunker in chunkers:
                            current_combination += 1
                            combination_start = datetime.datetime.now()
                            print(f"\n[{combination_start}] Starting combination ({current_combination}/{total_combinations})")
                            process_combination(parser, chunker, overlap, retriever, model, base_dir, all_questions, top_k, dataset)
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
    
    # Configuration lists - edit these to change what gets processed
    DATASETS = ["financebench", "tablequest"] # "financebench",
    PARSERS = ['pdfminer', 'pymupdf', 'pypdf2', 'unstructured', 'pdfplumber', 'pypdfium2']
    CHUNKERS = ['token', 'sentence', 'semantic', 'recursive', 'sdpm', 'neural']
    OVERLAPS = [128]
    RETRIEVERS = ["SpladeRetriever"]  # Add: "ColBERTRetriever", "SentenceTransformerRetriever", "SpladeRetriever"
    MODELS = ["llama3.2:3b", "deepseek-r1:1.5b"] #,"gemma3:4b", "phi3:mini"] 
    """
        gemma3:4b", "phi3:mini done for k=1 and k=3
        "llama3.2:3b", "deepseek-r1:1.5b" done top=1 and k=3
        "qwen3:4b" too slow
    """
    
    # Number of top pages to use as context
    TOP_K = 3
    
    # =============================================================================
    # END CONFIGURATION
    # =============================================================================
    
    print(f"Configuration:")
    print(f"  Parsers: {PARSERS}")
    print(f"  Chunkers: {CHUNKERS}")
    print(f"  Overlaps: {OVERLAPS}")
    print(f"  Retrievers: {RETRIEVERS}")
    print(f"  Models: {MODELS}")
    print(f"  Datasets: {DATASETS}")
    print(f"  Top-K pages: {TOP_K}")
    print(f"  Total combinations: {len(OVERLAPS) * len(RETRIEVERS) * len(MODELS) * len(PARSERS) * len(CHUNKERS) * len(DATASETS)}")
    
    # Validate setup
    if not check_paths(BASE_DIR, DATASETS):
        print("❌ Setup validation failed. Please fix the issues before running.")
        exit(1)
    
    print("✅ Setup validation passed. Starting processing...")
    
    # Run main processing
    main(PARSERS, CHUNKERS, OVERLAPS, RETRIEVERS, MODELS, BASE_DIR, TOP_K, DATASETS)
    
    script_end = datetime.datetime.now()
    overall_duration = script_end - script_start
    print(f"\n🎉 Script finished at: {script_end}")
    print(f"Total execution time: {overall_duration}")
    overall_duration = script_end - script_start
    print(f"\n🎉 Script finished at: {script_end}")
    print(f"Total execution time: {overall_duration}")
