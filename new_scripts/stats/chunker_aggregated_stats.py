#!/usr/bin/env python3
"""
Script to aggregate chunk statistics across all parsers for each chunking strategy and overlap setting.
Creates a summary table showing aggregate statistics for each chunker-overlap combination.
"""

import json
import os
import glob
from collections import defaultdict
from tabulate import tabulate
import argparse
import statistics


def load_chunk_file(json_path):
    """Load a chunk JSON file and return the number of chunks."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("chunks", []))
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return 0


def analyze_chunker_stats(chunks_base_dir):
    """
    Analyze chunk statistics aggregated across all parsers for each chunker-overlap combination.
    
    Args:
        chunks_base_dir: Base directory containing chunked data
        
    Returns:
        Dictionary with chunker-overlap combinations and their aggregate statistics
    """
    # Structure: {chunker-overlap: {parser: [chunks_per_file]}}
    chunker_data = defaultdict(lambda: defaultdict(list))
    
    # Get all parsers
    if not os.path.exists(chunks_base_dir):
        print(f"Directory {chunks_base_dir} does not exist!")
        return {}
    
    parsers = [d for d in os.listdir(chunks_base_dir) 
               if os.path.isdir(os.path.join(chunks_base_dir, d))]
    
    print(f"Found parsers: {parsers}")
    
    for parser in parsers:
        parser_path = os.path.join(chunks_base_dir, parser)
        
        # Get all chunkers for this parser
        chunkers = [d for d in os.listdir(parser_path) 
                   if os.path.isdir(os.path.join(parser_path, d))]
        
        print(f"  Processing chunkers for {parser}: {chunkers}")
        
        for chunker in chunkers:
            chunker_path = os.path.join(parser_path, chunker)
            
            # Get all overlap directories for this chunker
            overlap_dirs = [d for d in os.listdir(chunker_path) 
                           if os.path.isdir(os.path.join(chunker_path, d)) and d.startswith('overlap_')]
            
            for overlap_dir in overlap_dirs:
                overlap_path = os.path.join(chunker_path, overlap_dir)
                overlap_value = overlap_dir.replace('overlap_', '')
                chunker_overlap_key = f"{chunker}-{overlap_value}"
                
                # Get all JSON files in this overlap directory
                json_files = glob.glob(os.path.join(overlap_path, "*.json"))
                
                for json_file in json_files:
                    chunk_count = load_chunk_file(json_file)
                    if chunk_count > 0:  # Only include files with chunks
                        chunker_data[chunker_overlap_key][parser].append(chunk_count)
    
    return chunker_data


def calculate_aggregate_stats(chunker_data):
    """
    Calculate aggregate statistics for each chunker-overlap combination.
    
    Args:
        chunker_data: Dictionary with chunker-overlap combinations and parser data
        
    Returns:
        Dictionary with aggregate statistics
    """
    aggregate_stats = {}
    
    for chunker_overlap_key, parser_data in chunker_data.items():
        # Collect all chunk counts across all parsers
        all_chunks = []
        total_files = 0
        
        for parser, chunks_per_file in parser_data.items():
            all_chunks.extend(chunks_per_file)
            total_files += len(chunks_per_file)
        
        if all_chunks:
            aggregate_stats[chunker_overlap_key] = {
                'total_chunks': sum(all_chunks),
                'total_files': total_files,
                'avg_chunks_per_file': round(statistics.mean(all_chunks), 2),
                'min_chunks_per_file': min(all_chunks),
                'max_chunks_per_file': max(all_chunks),
                'median_chunks_per_file': round(statistics.median(all_chunks), 2),
                'std_chunks_per_file': round(statistics.stdev(all_chunks), 2) if len(all_chunks) > 1 else 0
            }
    
    return aggregate_stats


def create_aggregate_table(aggregate_stats):
    """Create a table showing aggregate statistics for each chunker-overlap combination."""
    headers = [
        "Chunker", 
        "Overlap", 
        "Total Chunks", 
        "Avg Chunks/Page", 
        "Min Chunks/Page", 
        "Max Chunks/Page"
    ]
    
    table_data = []
    
    for chunker_overlap_key, stats in sorted(aggregate_stats.items()):
        parts = chunker_overlap_key.split('-')
        if len(parts) >= 2:
            chunker = parts[0]
            overlap = parts[1]
        else:
            chunker = chunker_overlap_key
            overlap = "N/A"
        
        table_data.append([
            chunker,
            overlap,
            stats['total_chunks'],
            stats['avg_chunks_per_file'],
            stats['min_chunks_per_file'],
            stats['max_chunks_per_file']
        ])
    
    return tabulate(table_data, headers=headers, tablefmt="grid")


def create_detailed_table(aggregate_stats):
    """Create a detailed table with additional statistics."""
    headers = [
        "Chunker", 
        "Overlap", 
        "Total Files",
        "Total Chunks", 
        "Avg Chunks/Page", 
        "Min Chunks/Page", 
        "Max Chunks/Page",
        "Median Chunks/Page",
        "Std Dev"
    ]
    
    table_data = []
    
    for chunker_overlap_key, stats in sorted(aggregate_stats.items()):
        parts = chunker_overlap_key.split('-')
        if len(parts) >= 2:
            chunker = parts[0]
            overlap = parts[1]
        else:
            chunker = chunker_overlap_key
            overlap = "N/A"
        
        table_data.append([
            chunker,
            overlap,
            stats['total_files'],
            stats['total_chunks'],
            stats['avg_chunks_per_file'],
            stats['min_chunks_per_file'],
            stats['max_chunks_per_file'],
            stats['median_chunks_per_file'],
            stats['std_chunks_per_file']
        ])
    
    return tabulate(table_data, headers=headers, tablefmt="grid")


def main():
    dataset = 'financebench'  # Default dataset
    parser = argparse.ArgumentParser(description='Aggregate chunk statistics across all parsers for each chunking strategy')
    parser.add_argument('--chunks-dir', 
                       default=f'new_scripts/data/parsed_pages_chunks/{dataset}',
                       help='Base directory containing chunked data')
    parser.add_argument('--dataset',
                       default=dataset,
                       help='Dataset name (affects default chunks-dir)')
    parser.add_argument('--detailed',
                       action='store_true',
                       help='Show detailed statistics including median and standard deviation')
    
    args = parser.parse_args()
    
    # Update chunks_dir if dataset was specified
    if args.dataset != dataset:
        args.chunks_dir = f'new_scripts/data/parsed_pages_chunks/{args.dataset}'
    
    # Convert to absolute path if relative
    chunks_dir = os.path.abspath(args.chunks_dir) if not os.path.isabs(args.chunks_dir) else args.chunks_dir
    
    print(f"Analyzing chunker statistics in: {chunks_dir}")
    print("=" * 80)
    
    # Analyze the data
    chunker_data = analyze_chunker_stats(chunks_dir)
    
    if not chunker_data:
        print("No chunked data found!")
        return
    
    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_stats(chunker_data)
    
    if not aggregate_stats:
        print("No valid chunk statistics found!")
        return
    
    # Display results
    if args.detailed:
        print("\n📊 DETAILED CHUNKER STATISTICS (Aggregated Across All Parsers)")
        print("=" * 80)
        print(create_detailed_table(aggregate_stats))
    else:
        print("\n📈 CHUNKER STATISTICS (Aggregated Across All Parsers)")
        print("=" * 80)
        print(create_aggregate_table(aggregate_stats))
    
    # Print some overall statistics
    total_files = sum(stats['total_files'] for stats in aggregate_stats.values())
    total_chunks = sum(stats['total_chunks'] for stats in aggregate_stats.values())
    
    print(f"\n📋 OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total chunker-overlap combinations: {len(aggregate_stats)}")
    print(f"Total files processed: {total_files}")
    print(f"Total chunks generated: {total_chunks}")
    
    if total_files > 0:
        print(f"Average chunks per file (overall): {total_chunks / total_files:.2f}")
    
    # Show breakdown by chunker
    print(f"\n📊 CHUNKER BREAKDOWN")
    print("=" * 80)
    chunker_totals = defaultdict(lambda: {'files': 0, 'chunks': 0})
    
    for chunker_overlap_key, stats in aggregate_stats.items():
        chunker = chunker_overlap_key.split('-')[0]
        chunker_totals[chunker]['files'] += stats['total_files']
        chunker_totals[chunker]['chunks'] += stats['total_chunks']
    
    for chunker, totals in sorted(chunker_totals.items()):
        avg_chunks = totals['chunks'] / totals['files'] if totals['files'] > 0 else 0
        print(f"{chunker}: {totals['chunks']} chunks from {totals['files']} files (avg: {avg_chunks:.2f})")


if __name__ == '__main__':
    main()
