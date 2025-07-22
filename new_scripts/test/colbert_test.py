import os
import sys
sys.path.append(os.path.abspath(".."))

from retrievers.colbert import ColBERTRetriever 
from new_scripts.evaluation.classes.query_qrel_builder import QueryQrelsBuilder
from new_scripts.evaluation.classes.document_provider import DocumentProvider
import json
from pathlib import Path
import io
import contextlib

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET = "financebench"  # Change to your dataset name, e.g., "tablequest", "financebench"

# =============================================================================
# END CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    csv_path = f"new_scripts/data/chunks/{DATASET}/overlap_0/pdfminer_token_chunked_pages.csv"
    k_values = [1, 3, 5, 10]
    model_name = "lightonai/GTE-ModernColBERT-v1"  # "colbert-ir/colbertv2.0", "lightonai/GTE-ModernColBERT-v1"

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=False)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}") 

    retriever = ColBERTRetriever(provider, 
                                model_name=model_name,
                                index_folder="indexes/pylate-index", 
                                index_name="index", 
                                override=True,
                                batch_size=32,
                                device_map="cuda")
    run = retriever.search(queries, k=-1, agg='max')
    
    # Sort the run scores for each query in decreasing order
    sorted_run = {
        qid: dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        for qid, scores in run.items()
    }
    
    # Determine output directory based on csv_path and dataset
    csv_path_obj = Path(csv_path)
    out_dir = Path("new_scripts/data/retrieved_pages") / DATASET / csv_path_obj.parent.name / csv_path_obj.stem.replace("_chunked_pages", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = out_dir / f"{retriever.__class__.__name__}_run_sorted.json"
    with open(output_file, "w", encoding="utf-8") as outf:
        json.dump(sorted_run, outf, indent=2)
    print(f"✅ Saved sorted run scores → {output_file}")
    
    # Capture evaluation metrics output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        metrics = retriever.evaluate(run, qrels, k_values, verbose=True)
    metrics_output = f.getvalue()
    
    # Save evaluation results to text file
    results_dir = Path("new_scripts/data/retrieved_pages/ir_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"{retriever.__class__.__name__}_evaluation.txt"
    with open(results_file, "w", encoding="utf-8") as rf:
        rf.write("=== DATASET STATISTICS ===\n")
        rf.write(f"{provider.stats}\n")
        rf.write(f"Queries: {len(queries)}\n")
        rf.write(f"Qrels: {len(qrels)}\n\n")
        rf.write("=== USED MODEL ===\n")
        rf.write(f"Model: {model_name}\n\n")
        rf.write("=== EVALUATION METRICS ===\n")
        rf.write(metrics_output)
    
    print(f"✅ Saved evaluation results → {results_file}")
    
    # Print metrics to console as well
    print("\n=== EVALUATION METRICS ===")
    print(metrics_output)