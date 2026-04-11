#!/usr/bin/env python3
"""
Script to count unique PDFs in TableQuest sampled_pages_pdf directory.
Analyzes global stats and per question type (easy, medium, hard).
"""

import os
import re
from collections import defaultdict
from tabulate import tabulate


def extract_pdf_name(filename):
    """
    Extract PDF name from filename like '3M_2023Q2_10Q_p19.pdf'
    Returns '3M_2023Q2_10Q'
    """
    # Remove .pdf extension and extract everything before the last _p{number}
    base_name = filename.replace('.pdf', '')
    # Use regex to remove the _p{number} suffix
    match = re.match(r'^(.+)_p\d+$', base_name)
    if match:
        return match.group(1)
    else:
        # Fallback: just remove .pdf extension if pattern doesn't match
        return base_name


def analyze_pdf_directory(base_dir):
    """
    Analyze the sampled_pages_pdf directory and count unique PDFs.
    
    Args:
        base_dir: Path to the sampled_pages_pdf directory
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'by_difficulty': defaultdict(lambda: {'unique_pdfs': set(), 'total_files': 0}),
        'global_unique_pdfs': set(),
        'total_files': 0
    }
    
    difficulty_levels = ['easy', 'medium', 'hard']
    
    for difficulty in difficulty_levels:
        difficulty_path = os.path.join(base_dir, difficulty)
        
        if not os.path.exists(difficulty_path):
            print(f"Warning: Directory {difficulty_path} does not exist!")
            continue
            
        # Get all PDF files in this difficulty directory
        pdf_files = [f for f in os.listdir(difficulty_path) if f.endswith('.pdf')]
        
        stats['by_difficulty'][difficulty]['total_files'] = len(pdf_files)
        stats['total_files'] += len(pdf_files)
        
        # Extract unique PDF names
        for pdf_file in pdf_files:
            pdf_name = extract_pdf_name(pdf_file)
            stats['by_difficulty'][difficulty]['unique_pdfs'].add(pdf_name)
            stats['global_unique_pdfs'].add(pdf_name)
    
    return stats


def create_summary_table(stats):
    """Create a summary table of the statistics."""
    headers = ["Difficulty", "Total Files", "Unique PDFs", "Avg Pages per PDF"]
    table_data = []
    
    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in stats['by_difficulty']:
            total_files = stats['by_difficulty'][difficulty]['total_files']
            unique_pdfs = len(stats['by_difficulty'][difficulty]['unique_pdfs'])
            avg_pages = round(total_files / unique_pdfs, 2) if unique_pdfs > 0 else 0
            
            table_data.append([
                difficulty.capitalize(),
                total_files,
                unique_pdfs,
                avg_pages
            ])
    
    # Add total row
    total_unique = len(stats['global_unique_pdfs'])
    total_files = stats['total_files']
    avg_pages_global = round(total_files / total_unique, 2) if total_unique > 0 else 0
    
    table_data.append([
        "TOTAL",
        total_files,
        total_unique,
        avg_pages_global
    ])
    
    return tabulate(table_data, headers=headers, tablefmt="grid")


def show_pdf_overlap(stats):
    """Show which PDFs appear in multiple difficulty levels."""
    # Get PDFs for each difficulty
    easy_pdfs = stats['by_difficulty']['easy']['unique_pdfs']
    medium_pdfs = stats['by_difficulty']['medium']['unique_pdfs']
    hard_pdfs = stats['by_difficulty']['hard']['unique_pdfs']
    
    # Find overlaps
    easy_medium = easy_pdfs.intersection(medium_pdfs)
    easy_hard = easy_pdfs.intersection(hard_pdfs)
    medium_hard = medium_pdfs.intersection(hard_pdfs)
    all_three = easy_pdfs.intersection(medium_pdfs, hard_pdfs)
    
    print("\n📊 PDF OVERLAP ANALYSIS")
    print("=" * 50)
    print(f"PDFs in both Easy and Medium: {len(easy_medium)}")
    print(f"PDFs in both Easy and Hard: {len(easy_hard)}")
    print(f"PDFs in both Medium and Hard: {len(medium_hard)}")
    print(f"PDFs in all three difficulties: {len(all_three)}")
    
    if all_three:
        print(f"\nPDFs appearing in all difficulties:")
        for pdf in sorted(all_three):
            print(f"  - {pdf}")


def show_top_pdfs(stats):
    """Show PDFs with the most pages sampled."""
    pdf_page_counts = defaultdict(int)
    
    # Count pages for each PDF across all difficulties
    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in stats['by_difficulty']:
            difficulty_path = f"tablequest/sampled_pages_pdf/{difficulty}"
            if os.path.exists(difficulty_path):
                pdf_files = [f for f in os.listdir(difficulty_path) if f.endswith('.pdf')]
                for pdf_file in pdf_files:
                    pdf_name = extract_pdf_name(pdf_file)
                    pdf_page_counts[pdf_name] += 1
    
    # Sort by page count
    sorted_pdfs = sorted(pdf_page_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📈 TOP 10 PDFs BY NUMBER OF SAMPLED PAGES")
    print("=" * 50)
    headers = ["PDF Name", "Sampled Pages"]
    table_data = []
    
    for pdf_name, page_count in sorted_pdfs[:10]:
        table_data.append([pdf_name, page_count])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def show_all_unique_pdfs(stats):
    """Show all unique PDFs sorted alphabetically."""
    all_pdfs = sorted(stats['global_unique_pdfs'])
    
    print(f"\n📚 ALL UNIQUE PDFs ({len(all_pdfs)} total)")
    print("=" * 80)
    
    # Print in columns for better readability
    for i, pdf in enumerate(all_pdfs, 1):
        print(f"{i:3d}. {pdf}")


def quick_summary(stats):
    """Print a quick summary like the compact version."""
    print("\n📊 QUICK SUMMARY")
    print("-" * 40)
    
    for difficulty in ['easy', 'medium', 'hard']:
        if difficulty in stats['by_difficulty']:
            files = stats['by_difficulty'][difficulty]['total_files']
            unique = len(stats['by_difficulty'][difficulty]['unique_pdfs'])
            avg = files / unique if unique > 0 else 0
            print(f"{difficulty.capitalize():>6}: {files:>3} files, {unique:>2} unique PDFs ({avg:.1f} avg pages/PDF)")
    
    print("-" * 40)
    total_files = stats['total_files']
    total_unique = len(stats['global_unique_pdfs'])
    avg_global = total_files / total_unique if total_unique else 0
    print(f"{'Total':>6}: {total_files:>3} files, {total_unique:>2} unique PDFs ({avg_global:.1f} avg pages/PDF)")


def main():
    base_dir = "tablequest/sampled_pages_pdf"
    
    print(f"Analyzing unique PDFs in: {base_dir}")
    print("=" * 80)
    
    # Analyze the directory
    stats = analyze_pdf_directory(base_dir)
    
    # Show quick summary first (from compact script)
    quick_summary(stats)
    
    # Display detailed results
    print("\n📋 DETAILED SUMMARY STATISTICS")
    print("=" * 80)
    print(create_summary_table(stats))
    
    # Show overlap analysis
    show_pdf_overlap(stats)
    
    # Show top PDFs
    show_top_pdfs(stats)
    
    print(f"\n📊 GLOBAL STATISTICS")
    print("=" * 50)
    print(f"Total unique PDFs across all difficulties: {len(stats['global_unique_pdfs'])}")
    print(f"Total sampled pages: {stats['total_files']}")
    print(f"Average pages per PDF: {stats['total_files'] / len(stats['global_unique_pdfs']):.2f}")
    
    # Show all unique PDFs
    show_all_unique_pdfs(stats)


if __name__ == '__main__':
    main()
