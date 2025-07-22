#!/usr/bin/env python3
"""
Script to extract specific pages from PDFs based on PNG filenames.

This script:
1. Scans all PNG files in tablequest/sampled_pages/{easy,medium,hard}/
2. Extracts document name and page number from PNG filename
3. Finds corresponding PDF in tablequest/sampled_pdfs/
4. Extracts the specific page and saves as individual PDF
"""

import sys
import os
import re
from pathlib import Path
import fitz  # PyMuPDF for PDF manipulation

def parse_png_filename(png_path):
    """
    Parse PNG filename to extract document name and page number.
    
    Example: "JPMORGAN_2021Q1_10Q_p114.png" -> ("JPMORGAN_2021Q1_10Q", 114)
    """
    filename = png_path.stem
    
    # Pattern to match: {document_name}_p{page_number}
    match = re.match(r'^(.+)_p(\d+)$', filename)
    if match:
        document_name = match.group(1)
        page_number = int(match.group(2))
        return document_name, page_number
    else:
        print(f"Warning: Could not parse filename {filename}")
        return None, None

def find_matching_pdf(document_name, pdf_dir):
    """
    Find the PDF file that matches the document name.
    
    Args:
        document_name: The base document name from PNG filename
        pdf_dir: Directory containing PDFs
    
    Returns:
        Path to matching PDF or None if not found
    """
    # First try exact match
    exact_match = pdf_dir / f"{document_name}.pdf"
    if exact_match.exists():
        return exact_match
    
    # Try variations for common naming differences
    # Replace underscores with hyphens (common difference)
    alt_name = document_name.replace('_', '-')
    alt_match = pdf_dir / f"{alt_name}.pdf"
    if alt_match.exists():
        return alt_match
    
    # Try partial matching (in case of slight name differences)
    for pdf_file in pdf_dir.glob("*.pdf"):
        pdf_stem = pdf_file.stem
        # Check if document_name is a substring of the PDF name or vice versa
        if document_name in pdf_stem or pdf_stem in document_name:
            print(f"Found partial match: {document_name} -> {pdf_file.name}")
            return pdf_file
    
    print(f"Warning: No matching PDF found for {document_name}")
    return None

def extract_page_to_pdf(source_pdf, page_number, output_path):
    """
    Extract a specific page from a PDF and save as a new PDF.
    
    Args:
        source_pdf: Path to source PDF
        page_number: Page number to extract (1-indexed)
        output_path: Path where to save the extracted page
    """
    try:
        # Open the source PDF
        doc = fitz.open(source_pdf)
        
        # Check if page number is valid (convert to 0-indexed)
        page_index = page_number - 1
        if page_index < 0 or page_index >= doc.page_count:
            print(f"Error: Page {page_number} not found in {source_pdf} (has {doc.page_count} pages)")
            doc.close()
            return False
        
        # Create new PDF with just the target page
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the new PDF
        new_doc.save(str(output_path))
        new_doc.close()
        doc.close()
        
        print(f"✅ Extracted page {page_number} from {source_pdf.name} -> {output_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting page {page_number} from {source_pdf}: {e}")
        return False

def main():
    # Define paths
    base_dir = Path("../..")
    sampled_pages_dir = base_dir / "tablequest" / "sampled_pages"
    sampled_pdfs_dir = base_dir / "tablequest" / "sampled_pdfs"
    output_dir = base_dir / "tablequest" / "sampled_pages_pdf"
    
    print(f"📂 PNG source directory: {sampled_pages_dir}")
    print(f"📂 PDF source directory: {sampled_pdfs_dir}")
    print(f"📂 Output directory: {output_dir}")
    print()
    
    # Check if directories exist
    if not sampled_pages_dir.exists():
        print(f"❌ Error: PNG directory not found: {sampled_pages_dir}")
        return
    
    if not sampled_pdfs_dir.exists():
        print(f"❌ Error: PDF directory not found: {sampled_pdfs_dir}")
        return
    
    # Process each difficulty level
    difficulties = ["easy", "medium", "hard"]
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    for difficulty in difficulties:
        difficulty_dir = sampled_pages_dir / difficulty
        if not difficulty_dir.exists():
            print(f"⚠️  Directory not found: {difficulty_dir}")
            continue
        
        print(f"🚀 Processing {difficulty} difficulty...")
        
        # Create output subdirectory for this difficulty
        output_subdir = output_dir / difficulty
        
        # Process all PNG files in this difficulty
        png_files = list(difficulty_dir.glob("*.png"))
        print(f"Found {len(png_files)} PNG files in {difficulty}/")
        
        for png_file in png_files:
            total_processed += 1
            
            # Parse filename to get document name and page number
            document_name, page_number = parse_png_filename(png_file)
            if document_name is None:
                total_failed += 1
                continue
            
            # Find matching PDF
            pdf_file = find_matching_pdf(document_name, sampled_pdfs_dir)
            if pdf_file is None:
                total_failed += 1
                continue
            
            # Define output path
            output_filename = f"{png_file.stem}.pdf"  # Keep same name but change extension
            output_path = output_subdir / output_filename
            
            # Skip if already exists
            if output_path.exists():
                print(f"⏭️  Skipping {output_filename} (already exists)")
                total_success += 1
                continue
            
            # Extract the page
            success = extract_page_to_pdf(pdf_file, page_number, output_path)
            if success:
                total_success += 1
            else:
                total_failed += 1
        
        print(f"✅ Finished processing {difficulty} difficulty\n")
    
    # Print summary
    print("=" * 60)
    print("📈 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total PNG files processed: {total_processed}")
    print(f"Successfully extracted: {total_success}")
    print(f"Failed extractions: {total_failed}")
    print(f"Success rate: {(total_success/total_processed*100):.1f}%" if total_processed > 0 else "0%")
    print("=" * 60)

if __name__ == "__main__":
    main()
