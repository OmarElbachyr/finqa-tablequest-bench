import sys 
import os 
sys.path.append(os.path.abspath(".."))

from retrievers.bm25 import BM25Retriever
from new_scripts.evaluation.classes.document_provider import DocumentProvider
from new_scripts.evaluation.classes.query_qrel_builder import QueryQrelsBuilder
import json
from pathlib import Path
import io
import contextlib

 # =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================

# Configuration lists
PARSERS = ['pdfminer', 'pymupdf'] # 'pdfminer', 'pymupdf', 'pypdf2', 'unstructured', 'pdfplumber', 'pypdfium2'
CHUNKERS = ['token'] # 'token', 'sentence', 'semantic', 'recursive', 'sdpm', 'neural'
OVERLAPS = [0] # 0, 128, 256
DATASET = "financebench"  # Change to your dataset name, e.g., "tablequest", "financebench"

# Evaluation parameters
k_values = [1, 3, 5, 10]

# =============================================================================
# END CONFIGURATION
# =============================================================================

def process_combination(parser: str, chunker: str, overlap: int, k_values: list):
    """Process a single combination of parser, chunker, and overlap."""

    csv_path = f"new_scripts/data/chunks/{DATASET}/overlap_{overlap}/{parser}_{chunker}_chunked_pages.csv"
    
    # Check if CSV file exists
    if not Path(csv_path).exists():
        print(f"❌ Skipping {parser}_{chunker} overlap_{overlap} - CSV file not found: {csv_path}")
        return None
    
    print(f"\n🔄 Processing: {parser}_{chunker} | overlap_{overlap}")
    print(f"   CSV: {csv_path}")
    
    try:
        provider = DocumentProvider(csv_path, use_nltk_preprocessor=False)
        print(f"   {provider.stats}")
        
        queries, qrels = QueryQrelsBuilder(csv_path).build()
        print(f"   Queries: {len(queries)}, Qrels: {len(qrels)}")      

        bm25 = BM25Retriever(provider)
        run = bm25.search(queries, agg="max")  # max, mean, sum

        # Sort the run scores for each query in decreasing order
        sorted_run = {
            qid: dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
            for qid, scores in run.items()
        }

        # Determine output directory based on csv_path and dataset
        csv_path_obj = Path(csv_path)
        out_dir = Path("new_scripts/data/retrieved_pages") / DATASET / csv_path_obj.parent.name / csv_path_obj.stem.replace("_chunked_pages", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = out_dir / f"{bm25.__class__.__name__}_run_sorted.json"
        with open(output_file, "w", encoding="utf-8") as outf:
            json.dump(sorted_run, outf, indent=2)
        print(f"   ✅ Saved sorted run scores → {output_file}")

        # Capture evaluation metrics output
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            metrics = bm25.evaluate(run, qrels, k_values, verbose=True)
        metrics_output = f.getvalue()
        
        # Save the detailed table format
        tables_dir = Path("new_scripts/data/retrieved_pages/ir_results") / DATASET / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        table_file = tables_dir / f"{parser}_{chunker}_overlap_{overlap}_{bm25.__class__.__name__}_evaluation.txt"
        with open(table_file, "w", encoding="utf-8") as tf:
            tf.write(f"=== CONFIGURATION ===\n")
            tf.write(f"Dataset: {DATASET}\n")
            tf.write(f"Parser: {parser}\n")
            tf.write(f"Chunker: {chunker}\n")
            tf.write(f"Overlap: {overlap}\n")
            tf.write(f"Retriever: {bm25.__class__.__name__}\n")
            tf.write(f"CSV: {csv_path}\n\n")
            tf.write("=== DATASET STATISTICS ===\n")
            tf.write(f"{provider.stats}\n")
            tf.write(f"Queries: {len(queries)}\n")
            tf.write(f"Qrels: {len(qrels)}\n\n")
            tf.write("=== EVALUATION METRICS ===\n")
            tf.write(metrics_output)
        
        print(f"   ✅ Saved table format results → {table_file}")
        
        # Return complete metrics summary for consolidated JSON
        result = {
            "parser": parser,
            "chunker": chunker,
            "overlap": overlap,
            "num_queries": len(queries),
            "num_chunks": provider.stats.get('total_chunks', 0),
            "metrics": {}
        }
        
        # Add all k-value metrics
        for k in k_values:
            if k in metrics:
                result["metrics"][f"k{k}"] = {
                    "ndcg": round(metrics[k].get("ndcg", 0.0), 4),
                    "precision": round(metrics[k].get("precision", 0.0), 4),
                    "recall": round(metrics[k].get("recall", 0.0), 4)
                }
        
        # Add global metrics
        if "global" in metrics:
            result["metrics"]["global"] = {
                "mrr": round(metrics["global"].get("mrr", 0.0), 4),
                "rprec": round(metrics["global"].get("rprec", 0.0), 4)
            }
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error processing {parser}_{chunker} overlap_{overlap}: {str(e)}")
        return None


if __name__ == "__main__": 
    print("🚀 BM25 Retrieval Evaluation Script")
    print("="*50)
    print(f"Configuration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Parsers: {PARSERS}")
    print(f"  Chunkers: {CHUNKERS}")
    print(f"  Overlaps: {OVERLAPS}")
    print(f"  K-values: {k_values}")
    
    total_combinations = len(PARSERS) * len(CHUNKERS) * len(OVERLAPS)
    print(f"  Total combinations: {total_combinations}")
    print("="*50)
    
    # Process all combinations
    current_combination = 0
    all_results = []
    
    for parser in PARSERS:
        for chunker in CHUNKERS:
            for overlap in OVERLAPS:
                current_combination += 1
                print(f"\n[{current_combination}/{total_combinations}] Processing combination:")
                result = process_combination(parser, chunker, overlap, k_values)
                if result:
                    all_results.append(result)
    
    # Save consolidated results to single JSON file
    if all_results:
        consolidated_results = {
            "experiment_config": {
                "dataset": DATASET,
                "parsers": PARSERS,
                "chunkers": CHUNKERS,
                "overlaps": OVERLAPS,
                "k_values": k_values,
                "retriever": "BM25Retriever",
                "total_combinations": total_combinations
            },
            "results": all_results
        }
        
        json_dir = Path("new_scripts/data/retrieved_pages/ir_results") / DATASET / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        
        consolidated_file = json_dir / f"bm25_evaluation_summary_{DATASET}.json"
        with open(consolidated_file, "w", encoding="utf-8") as cf:
            json.dump(consolidated_results, cf, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved consolidated results → {consolidated_file}")
        
        # Calculate and display average metrics across all combinations
        print(f"\n📊 Calculating average metrics across {len(all_results)} combinations...")
        
        # Initialize metric accumulators
        avg_metrics = {}
        k_values_found = set()
        
        # Collect all metrics
        for result in all_results:
            metrics = result.get("metrics", {})
            for k_key, k_metrics in metrics.items():
                if k_key.startswith("k"):
                    k_values_found.add(k_key)
                    if k_key not in avg_metrics:
                        avg_metrics[k_key] = {"ndcg": [], "precision": [], "recall": []}
                    
                    avg_metrics[k_key]["ndcg"].append(k_metrics.get("ndcg", 0.0))
                    avg_metrics[k_key]["precision"].append(k_metrics.get("precision", 0.0))
                    avg_metrics[k_key]["recall"].append(k_metrics.get("recall", 0.0))
                elif k_key == "global":
                    if "global" not in avg_metrics:
                        avg_metrics["global"] = {"mrr": [], "rprec": []}
                    
                    avg_metrics["global"]["mrr"].append(k_metrics.get("mrr", 0.0))
                    avg_metrics["global"]["rprec"].append(k_metrics.get("rprec", 0.0))
        
        # Calculate averages
        summary_table = []
        summary_table.append("=" * 80)
        summary_table.append(f"BM25 RETRIEVAL EVALUATION - AVERAGE METRICS SUMMARY ({DATASET.upper()})")
        summary_table.append("=" * 80)
        summary_table.append(f"Experiment Configuration:")
        summary_table.append(f"  Dataset: {DATASET}")
        summary_table.append(f"  Parsers: {PARSERS}")
        summary_table.append(f"  Chunkers: {CHUNKERS}")
        summary_table.append(f"  Overlaps: {OVERLAPS}")
        summary_table.append(f"  Total Combinations: {len(all_results)}")
        summary_table.append("")
        summary_table.append("AVERAGE METRICS ACROSS ALL COMBINATIONS:")
        summary_table.append("=" * 60)
        
        # Sort k-values for consistent ordering
        sorted_k_values = sorted([k for k in k_values_found if k.startswith("k")], 
                                key=lambda x: int(x[1:]))
        
        for k_key in sorted_k_values:
            k_num = k_key[1:]  # Remove 'k' prefix
            if k_key in avg_metrics:
                avg_ndcg = sum(avg_metrics[k_key]["ndcg"]) / len(avg_metrics[k_key]["ndcg"])
                avg_precision = sum(avg_metrics[k_key]["precision"]) / len(avg_metrics[k_key]["precision"])
                avg_recall = sum(avg_metrics[k_key]["recall"]) / len(avg_metrics[k_key]["recall"])
                
                summary_table.append(f"K={k_num:<3} NDCG:{avg_ndcg:.4f}  P:{avg_precision:.4f}  R:{avg_recall:.4f}")
        
        if "global" in avg_metrics:
            avg_mrr = sum(avg_metrics["global"]["mrr"]) / len(avg_metrics["global"]["mrr"])
            avg_rprec = sum(avg_metrics["global"]["rprec"]) / len(avg_metrics["global"]["rprec"])
            summary_table.append(f"GLOBAL MRR:{avg_mrr:.4f}  Rprec:{avg_rprec:.4f}")
        
        summary_table.append("")
        summary_table.append("INDIVIDUAL COMBINATION RESULTS:")
        summary_table.append("=" * 60)
        
        for i, result in enumerate(all_results, 1):
            summary_table.append(f"{i}. {result['parser']}_{result['chunker']} (overlap_{result['overlap']}):")
            summary_table.append(f"   Queries: {result['num_queries']}, Chunks: {result['num_chunks']}")
            
            metrics = result.get("metrics", {})
            for k_key in sorted_k_values:
                if k_key in metrics:
                    k_num = k_key[1:]
                    k_metrics = metrics[k_key]
                    summary_table.append(f"   K={k_num:<3} NDCG:{k_metrics['ndcg']:.4f}  P:{k_metrics['precision']:.4f}  R:{k_metrics['recall']:.4f}")
            
            if "global" in metrics:
                global_metrics = metrics["global"]
                summary_table.append(f"   GLOBAL MRR:{global_metrics['mrr']:.4f}  Rprec:{global_metrics['rprec']:.4f}")
            summary_table.append("")
        
        summary_table.append("=" * 80)
        
        # Print summary to console
        summary_text = "\n".join(summary_table)
        print(summary_text)
        
        # Save summary to file
        summary_file = Path("new_scripts/data/retrieved_pages/ir_results") / DATASET / f"bm25_average_metrics_summary_{DATASET}.txt"
        with open(summary_file, "w", encoding="utf-8") as sf:
            sf.write(summary_text)
        
        print(f"\n✅ Saved average metrics summary → {summary_file}")
    
    print(f"\n🎉 Completed processing {total_combinations} combinations!")
    print(f"Check 'new_scripts/data/retrieved_pages/{DATASET}/' for sorted run files")
    print(f"Check 'new_scripts/data/retrieved_pages/ir_results/{DATASET}/json/bm25_evaluation_summary_{DATASET}.json' for consolidated results")
    print(f"Check 'new_scripts/data/retrieved_pages/ir_results/{DATASET}/tables/' for detailed table metrics")
    print(f"Check 'new_scripts/data/retrieved_pages/ir_results/{DATASET}/bm25_average_metrics_summary_{DATASET}.txt' for average metrics summary")
