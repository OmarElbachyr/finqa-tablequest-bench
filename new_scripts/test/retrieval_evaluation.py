import sys 
import os 
sys.path.append(os.path.abspath(".."))

from retrievers.bm25 import BM25Retriever
from retrievers.colbert import ColBERTRetriever
from retrievers.sentence_transformer import SentenceTransformerRetriever
from retrievers.splade import SpladeRetriever

from new_scripts.evaluation.classes.document_provider import DocumentProvider
from new_scripts.evaluation.classes.query_qrel_builder import QueryQrelsBuilder
import json
from pathlib import Path
import io
import contextlib
import time

# Try to import torch for device detection, fallback to CPU if not available
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

# =============================================================================
# CONFIGURATION - Edit these variables as needed
# =============================================================================

# Configuration lists
PARSERS = ['pdfminer', 'pymupdf', 'pypdf2', 'unstructured', 'pdfplumber', 'pypdfium2'] # 'pdfminer', 'pymupdf', 'pypdf2', 'unstructured', 'pdfplumber', 'pypdfium2'
CHUNKERS = ['token', 'sentence', 'semantic', 'recursive', 'sdpm', 'neural'] # 'token', 'sentence', 'semantic', 'recursive', 'sdpm', 'neural'
OVERLAPS = [0, 256] # 0, 128, 256
RETRIEVERS = ['BM25Retriever', 'ColBERTRetriever', 'SentenceTransformerRetriever', 'SpladeRetriever'] # 'BM25Retriever', 'ColBERTRetriever', 'SentenceTransformerRetriever', 'SpladeRetriever'
DATASET = "tablequest"  # Change to your dataset name, e.g., "tablequest", "financebench"

# =============================================================================
# RETRIEVER CONFIGURATION - Customize retriever initialization and search parameters
# =============================================================================
# Configuration based on arguments used in the test scripts:
# 
# BM25Retriever:
#   - No additional init args (only takes provider)
#   - search_args: agg ('max', 'mean', 'sum')
#
# ColBERTRetriever:
#   - init_args: model_name, device_map, batch_size, index_folder, index_name, override
#   - search_args: k (-1 for all documents), agg
#
# SentenceTransformerRetriever:
#   - init_args: model_name, device_map, is_instruct, task_description
#   - search_args: agg
#
# SpladeRetriever:
#   - init_args: model_name, device, batch_size, k_tokens_index
#   - search_args: agg

# Retriever initialization parameters - specify custom arguments here
RETRIEVER_CONFIGS = {
    'BM25Retriever': {
        'init_args': {},  # BM25Retriever only takes provider (no additional args)
        'search_args': {'agg': 'max'}  # Search aggregation: 'max', 'mean', 'sum'
    },
    'ColBERTRetriever': {
        'init_args': {
            'model_name': 'lightonai/GTE-ModernColBERT-v1',  # or 'colbert-ir/colbertv2.0'
            'device_map': 'cuda' if CUDA_AVAILABLE else 'cpu',
            'batch_size': 32,
            'index_folder': 'indexes/pylate-index',
            'index_name': 'index',
            'override': True
        },
        'search_args': {
            'k': -1,  # -1 for all documents
            'agg': 'max'
        }
    },
    'SentenceTransformerRetriever': {
        'init_args': {
            'model_name': 'intfloat/multilingual-e5-large',  # or 'BAAI/bge-m3', 'sentence-transformers/all-MiniLM-L6-v2'
            'device_map': 'cuda' if CUDA_AVAILABLE else 'cpu',
            'is_instruct': False,
            'task_description': 'Given a user query, retrieve the most relevant passages from the document corpus'
        },
        'search_args': {
            'agg': 'max'
        }
    },
    'SpladeRetriever': {
        'init_args': {
            'model_name': 'naver/splade-v3',
            'device': 'cuda' if CUDA_AVAILABLE else 'cpu',
            'batch_size': 16,
            'k_tokens_index': 256
        },
        'search_args': {
            'agg': 'max'
        }
    }
}

# Evaluation parameters
k_values = [1, 3, 5, 10]

# =============================================================================
# END CONFIGURATION
# =============================================================================

def display_retriever_configurations():
    """Display the current retriever configurations."""
    print("\n🔧 Retriever Configurations:")
    print("=" * 50)
    
    for retriever_name in RETRIEVERS:
        if retriever_name in RETRIEVER_CONFIGS:
            config = RETRIEVER_CONFIGS[retriever_name]
            print(f"\n{retriever_name}:")
            
            init_args = config.get('init_args', {})
            if init_args:
                print(f"  Init args: {init_args}")
            else:
                print(f"  Init args: (default)")
            
            search_args = config.get('search_args', {})
            if search_args:
                print(f"  Search args: {search_args}")
            else:
                print(f"  Search args: (default)")
        else:
            print(f"\n{retriever_name}: (no custom configuration)")
    
    print("=" * 50)


def get_retriever_instance(retriever_name: str, provider):
    """Get an initialized retriever instance with custom arguments."""
    retriever_map = {
        'BM25Retriever': BM25Retriever,
        'ColBERTRetriever': ColBERTRetriever,
        'SentenceTransformerRetriever': SentenceTransformerRetriever,
        'SpladeRetriever': SpladeRetriever,
    }
    
    if retriever_name not in retriever_map:
        raise ValueError(f"Unknown retriever: {retriever_name}. Available: {list(retriever_map.keys())}")
    
    RetrieverClass = retriever_map[retriever_name]
    
    # Get initialization arguments from configuration
    config = RETRIEVER_CONFIGS.get(retriever_name, {})
    init_args = config.get('init_args', {})
    
    # Initialize retriever with provider and custom arguments
    retriever = RetrieverClass(provider, **init_args)
    
    return retriever


def process_combination(parser: str, chunker: str, overlap: int, retriever_name: str, k_values: list):
    """Process a single combination of parser, chunker, overlap, and retriever."""
    csv_path = f"new_scripts/data/chunks/{DATASET}/overlap_{overlap}/{parser}_{chunker}_chunked_pages.csv"
    
    # Check if CSV file exists
    if not Path(csv_path).exists():
        print(f"❌ Skipping {parser}_{chunker} overlap_{overlap} {retriever_name} - CSV file not found: {csv_path}")
        return None
    
    print(f"\n🔄 Processing: {parser}_{chunker} | overlap_{overlap} | {retriever_name}")
    print(f"   CSV: {csv_path}")
    
    try:
        provider = DocumentProvider(csv_path, use_nltk_preprocessor=False)
        print(f"   {provider.stats}")
        
        queries, qrels = QueryQrelsBuilder(csv_path).build()
        print(f"   Queries: {len(queries)}, Qrels: {len(qrels)}")      

        # Measure indexing time
        start_index_time = time.time()
        retriever = get_retriever_instance(retriever_name, provider)
        end_index_time = time.time()
        indexing_time = end_index_time - start_index_time
        print(f"   ⏱️ Indexing time: {indexing_time:.2f} seconds")

        # Get search arguments from configuration
        config = RETRIEVER_CONFIGS.get(retriever_name, {})
        search_args = config.get('search_args', {})
        
        # Search with retriever using custom arguments
        run = retriever.search(queries, **search_args)

        # Sort the run scores for each query in decreasing order
        sorted_run = {
            qid: dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
            for qid, scores in run.items()
        }

        # Determine output directory based on csv_path and dataset
        csv_path_obj = Path(csv_path)
        out_dir = Path("new_scripts/data/retrieved_pages") / DATASET / csv_path_obj.parent.name / csv_path_obj.stem.replace("_chunked_pages", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = out_dir / f"{retriever_name}_run_sorted.json"
        with open(output_file, "w", encoding="utf-8") as outf:
            json.dump(sorted_run, outf, indent=2)
        print(f"   ✅ Saved sorted run scores → {output_file}")

        # Capture evaluation metrics output
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            metrics = retriever.evaluate(run, qrels, k_values, verbose=True)
        metrics_output = f.getvalue()
        
        # Save the detailed table format
        tables_dir = Path("new_scripts/data/retrieved_pages/ir_results") / DATASET / f"overlap_{overlap}" / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        
        table_file = tables_dir / f"{parser}_{chunker}_{retriever_name}_evaluation.txt"
        with open(table_file, "w", encoding="utf-8") as tf:
            tf.write(f"=== CONFIGURATION ===\n")
            tf.write(f"Dataset: {DATASET}\n")
            tf.write(f"Parser: {parser}\n")
            tf.write(f"Chunker: {chunker}\n")
            tf.write(f"Overlap: {overlap}\n")
            tf.write(f"Retriever: {retriever_name}\n")
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
            "retriever": retriever_name,
            "num_queries": len(queries),
            "num_chunks": provider.stats.get('total_chunks', 0),
            "indexing_time": round(indexing_time, 2),  # <-- add this line
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
        print(f"   ❌ Error processing {parser}_{chunker} overlap_{overlap} {retriever_name}: {str(e)}")
        return None


if __name__ == "__main__": 
    print("🚀 Generic Retrieval Evaluation Script")
    print("="*50)
    print(f"Configuration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Parsers: {PARSERS}")
    print(f"  Chunkers: {CHUNKERS}")
    print(f"  Overlaps: {OVERLAPS}")
    print(f"  Retrievers: {RETRIEVERS}")
    print(f"  K-values: {k_values}")
    
    total_combinations = len(PARSERS) * len(CHUNKERS) * len(OVERLAPS) * len(RETRIEVERS)
    print(f"  Total combinations: {total_combinations}")
    
    # Display retriever configurations
    display_retriever_configurations()
    
    print("="*50)
    
    # Process all combinations
    current_combination = 0
    all_results = []
    
    for parser in PARSERS:
        for chunker in CHUNKERS:
            for overlap in OVERLAPS:
                for retriever_name in RETRIEVERS:
                    current_combination += 1
                    print(f"\n[{current_combination}/{total_combinations}] Processing combination:")
                    result = process_combination(parser, chunker, overlap, retriever_name, k_values)
                    if result:
                        all_results.append(result)
    
    # Save consolidated results organized by overlap
    if all_results:
        # Group results by overlap
        overlap_results = {}
        for result in all_results:
            overlap = result["overlap"]
            if overlap not in overlap_results:
                overlap_results[overlap] = []
            overlap_results[overlap].append(result)
        
        # Save separate consolidated files for each overlap
        for overlap, overlap_data in overlap_results.items():
            consolidated_results = {
                "experiment_config": {
                    "dataset": DATASET,
                    "parsers": PARSERS,
                    "chunkers": CHUNKERS,
                    "overlap": overlap,
                    "retrievers": RETRIEVERS,
                    "k_values": k_values,
                    "total_combinations": len(overlap_data)
                },
                "results": overlap_data
            }
            
            json_dir = Path("new_scripts/data/retrieved_pages/ir_results") / DATASET / f"overlap_{overlap}" / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            
            consolidated_file = json_dir / f"retrieval_evaluation_summary_{DATASET}_overlap_{overlap}.json"
            with open(consolidated_file, "w", encoding="utf-8") as cf:
                json.dump(consolidated_results, cf, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Saved consolidated results for overlap_{overlap} → {consolidated_file}")
        
        # Calculate and display average metrics for each overlap separately
        print(f"\n📊 Calculating average metrics for each overlap...")
        
        # Group results by overlap
        overlap_results = {}
        for result in all_results:
            overlap = result["overlap"]
            if overlap not in overlap_results:
                overlap_results[overlap] = []
            overlap_results[overlap].append(result)
        
        # Generate summary for each overlap
        for overlap in sorted(overlap_results.keys()):
            overlap_data = overlap_results[overlap]
            print(f"\n📊 Processing overlap_{overlap} with {len(overlap_data)} combinations...")
            
            # Initialize metric accumulators grouped by retriever for this overlap
            retriever_metrics = {}
            
            # Collect all metrics grouped by retriever for this overlap
            for result in overlap_data:
                retriever_name = result["retriever"]
                if retriever_name not in retriever_metrics:
                    retriever_metrics[retriever_name] = {"results": [], "avg_metrics": {}}
                
                retriever_metrics[retriever_name]["results"].append(result)
                
                metrics = result.get("metrics", {})
                if "avg_metrics" not in retriever_metrics[retriever_name]:
                    retriever_metrics[retriever_name]["avg_metrics"] = {}
                
                for k_key, k_metrics in metrics.items():
                    if k_key.startswith("k"):
                        if k_key not in retriever_metrics[retriever_name]["avg_metrics"]:
                            retriever_metrics[retriever_name]["avg_metrics"][k_key] = {"ndcg": [], "precision": [], "recall": []}
                        
                        retriever_metrics[retriever_name]["avg_metrics"][k_key]["ndcg"].append(k_metrics.get("ndcg", 0.0))
                        retriever_metrics[retriever_name]["avg_metrics"][k_key]["precision"].append(k_metrics.get("precision", 0.0))
                        retriever_metrics[retriever_name]["avg_metrics"][k_key]["recall"].append(k_metrics.get("recall", 0.0))
                    elif k_key == "global":
                        if "global" not in retriever_metrics[retriever_name]["avg_metrics"]:
                            retriever_metrics[retriever_name]["avg_metrics"]["global"] = {"mrr": [], "rprec": []}
                        
                        retriever_metrics[retriever_name]["avg_metrics"]["global"]["mrr"].append(k_metrics.get("mrr", 0.0))
                        retriever_metrics[retriever_name]["avg_metrics"]["global"]["rprec"].append(k_metrics.get("rprec", 0.0))
            
            # Calculate averages and build summary for this overlap
            summary_table = []
            summary_table.append("=" * 80)
            summary_table.append(f"RETRIEVAL EVALUATION - AVERAGE METRICS SUMMARY ({DATASET.upper()} - OVERLAP_{overlap})")
            summary_table.append("=" * 80)
            summary_table.append(f"Experiment Configuration:")
            summary_table.append(f"  Dataset: {DATASET}")
            summary_table.append(f"  Parsers: {PARSERS}")
            summary_table.append(f"  Chunkers: {CHUNKERS}")
            summary_table.append(f"  Overlap: {overlap}")
            summary_table.append(f"  Retrievers: {RETRIEVERS}")
            summary_table.append(f"  Total Combinations for this overlap: {len(overlap_data)}")
            summary_table.append("")
            
            # For each retriever, show average metrics for this overlap
            for retriever_name, retriever_data in retriever_metrics.items():
                avg_metrics = retriever_data["avg_metrics"]
                num_combinations = len(retriever_data["results"])
                
                summary_table.append(f"RETRIEVER: {retriever_name} (Averaged across {num_combinations} combinations)")
                summary_table.append("=" * 60)
                
                # Sort k-values for consistent ordering
                k_values_found = [k for k in avg_metrics.keys() if k.startswith("k")]
                sorted_k_values = sorted(k_values_found, key=lambda x: int(x[1:]))
                
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
            
            for i, result in enumerate(overlap_data, 1):
                summary_table.append(f"{i}. {result['parser']}_{result['chunker']} | {result['retriever']}:")
                summary_table.append(f"   Queries: {result['num_queries']}, Chunks: {result['num_chunks']}")
                
                metrics = result.get("metrics", {})
                k_values_found = [k for k in metrics.keys() if k.startswith("k")]
                sorted_k_values = sorted(k_values_found, key=lambda x: int(x[1:]))
                
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
            
            # Save summary to file with overlap-specific name
            summary_file = Path("new_scripts/data/retrieved_pages/ir_results") / DATASET / f"overlap_{overlap}" / f"retrieval_average_metrics_summary_{DATASET}_overlap_{overlap}.txt"
            with open(summary_file, "w", encoding="utf-8") as sf:
                sf.write(summary_text)
            
            print(f"\n✅ Saved overlap_{overlap} metrics summary → {summary_file}")
        
        print(f"\n📊 Generated separate summaries for {len(overlap_results)} overlap values.")
    
    print(f"\n🎉 Completed processing {total_combinations} combinations!")
    print(f"Check 'new_scripts/data/retrieved_pages/{DATASET}/' for sorted run files")
    print(f"Check 'new_scripts/data/retrieved_pages/ir_results/{DATASET}/overlap_*/json/' for consolidated results")
    print(f"Check 'new_scripts/data/retrieved_pages/ir_results/{DATASET}/overlap_*/tables/' for detailed table metrics")
    print(f"Check 'new_scripts/data/retrieved_pages/ir_results/{DATASET}/overlap_*/' for overlap-specific average metrics summaries")
