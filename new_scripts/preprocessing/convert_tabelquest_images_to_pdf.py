#!/usr/bin/env python3
"""
Script to convert image files (PNG, JPG, JPEG) to PDF format.
This script processes all images in the TableQuest sampled_pages directory
and converts them to PDFs while preserving the directory structure.
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image

def convert_image_to_pdf(image_path: Path, output_path: Path) -> bool:
    """
    Convert a single image file to PDF format.
    
    Args:
        image_path: Path to the input image file
        output_path: Path where the PDF will be saved
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Open and convert image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as PDF
            img.save(output_path, 'PDF', resolution=100.0)
        
        return True
    except Exception as e:
        print(f"Error converting {image_path}: {e}")
        return False

def main():
    # Define paths
    input_dir = Path("tablequest/sampled_pages")
    output_dir = Path("tablequest/sampled_pages_pdf")
    
    print(f"🔄 Converting images from: {input_dir}")
    print(f"📁 Saving PDFs to: {output_dir}")
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"**/*{ext}")))
    
    total_files = len(image_files)
    print(f"📊 Found {total_files} image files to convert\n")
    
    if total_files == 0:
        print("❌ No image files found. Please check the input directory.")
        return
    
    # Track conversion stats
    successful_conversions = 0
    failed_conversions = 0
    start_time = time.time()
    
    for idx, image_path in enumerate(image_files, 1):
        # Calculate relative path to preserve directory structure
        rel_path = image_path.relative_to(input_dir)
        
        # Create output path with .pdf extension
        output_path = output_dir / rel_path.with_suffix('.pdf')
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Progress indicator
        print(f"[{idx}/{total_files}] Converting: {rel_path}")
        
        # Convert image to PDF
        if convert_image_to_pdf(image_path, output_path):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("📈 CONVERSION SUMMARY")
    print("="*60)
    print(f"✅ Successful conversions: {successful_conversions}")
    print(f"❌ Failed conversions: {failed_conversions}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Average time per file: {total_time/total_files:.3f} seconds")
    print(f"📁 Output directory: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
