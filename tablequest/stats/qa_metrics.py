import os
import json
from pathlib import Path
import tiktoken

def compute_metrics(json_path, dataset="default"):
    metrics = {}
    encoding = tiktoken.get_encoding("cl100k_base")

    if dataset == "financebench":
        # Treat json_path as the full file path
        financebench_path = Path(json_path)
        if not financebench_path.exists():
            print(f"Warning: {financebench_path} not found.")
            return metrics
        with financebench_path.open("r", encoding="utf-8") as f:
            doc_list = json.load(f)
        qa_pairs = []
        for doc in doc_list:
            qa_pairs.extend(doc.get("qa_pairs", []))
        question_lengths = [len(encoding.encode(pair["question"])) for pair in qa_pairs]
        answer_lengths = [len(encoding.encode(pair["answer"])) for pair in qa_pairs]
        qa_count = len(qa_pairs)
        metrics["financebench"] = {
            "Avg. Question Length": sum(question_lengths) / qa_count if qa_count > 0 else 0,
            "Avg. Answer Length": sum(answer_lengths) / qa_count if qa_count > 0 else 0,
            "QA Pairs Count": qa_count,
        }
        return metrics

    for level in ("easy", "medium", "hard"):
        json_file = Path(json_path) / f"adapted_{level}.json"
        if not json_file.exists():
            print(f"Warning: {json_file} not found.")
            continue
        with json_file.open("r", encoding="utf-8") as f:
            doc_list = json.load(f)
        qa_pairs = []
        for doc in doc_list:
            qa_pairs.extend(doc.get("qa_pairs", []))
        question_lengths = [len(encoding.encode(pair["question"])) for pair in qa_pairs]
        answer_lengths = [len(encoding.encode(pair["answer"])) for pair in qa_pairs]
        qa_count = len(qa_pairs)
        metrics[level] = {
            "Avg. Question Length": sum(question_lengths) / qa_count if qa_count > 0 else 0,
            "Avg. Answer Length": sum(answer_lengths) / qa_count if qa_count > 0 else 0,
            "QA Pairs Count": qa_count,
        }
    return metrics

if __name__ == "__main__":
    # Set dataset variable here
    dataset = "financebench"  # or "default"
    if dataset == "financebench":
        json_path = "new_scripts/data/csv/document_qa_pairs.json"  # Use exact file path
    else:
        json_path = "tablequest/qa_pairs/adapted_pairs"
    metrics = compute_metrics(json_path, dataset=dataset)
    for level, stats in metrics.items():
        print(f"\nMetrics for {level} difficulty:")
        for metric, value in stats.items():
            print(f"{metric}: {value}")
