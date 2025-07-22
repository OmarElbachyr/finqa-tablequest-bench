from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple, Set, Dict
import glob

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
JSON_PATH = Path("new_scripts/data/csv/tq_document_qa_pairs.json")
CHUNKS_BASE_DIR = Path("new_scripts/data/parsed_pages_chunks/tablequest")
OUTPUT_DIR = Path("new_scripts/data/chunks/tablequest")
LIMIT = -1  # set e.g. 3 for a quick test

# Output columns
COLUMNS = [
    "query",
    "document",
    "answer",
    "page_number",
    "chunk_type",       # always "text" here
    "text_description", # chunk content
    "chunk_id",         # document_page_index
    "image_filename",   # document_name_page_number
]

# Available parsers and chunkers
PARSERS = ['pdfminer', 'pymupdf', 'pypdf2', 'unstructured', 'pdfplumber', 'pypdfium2'] # 'pdfminer', 'pymupdf', 'pypdf2', 'unstructured', 'pdfplumber', 'pypdfium2'
CHUNKERS = ['token', 'sentence', 'semantic', 'recursive', 'sdpm', 'neural'] # 'token', 'sentence', 'semantic', 'recursive', 'sdpm', 'neural'
OVERLAP_SIZES = [0, 128, 256] # 0, 128, 256


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _collect_pages_to_extract(records) -> Dict[str, Set[int]]:
    """Collect all unique pages that need to be extracted per document."""
    pages_to_extract: Dict[str, Set[int]] = {}
    for doc_record in records:
        doc_name = doc_record["doc_name"]
        pages = set()
        for qa in doc_record["qa_pairs"]:
            pages.update(qa["evidence_pages"])
        if pages:
            # Merge pages if document already exists, otherwise create new entry
            if doc_name in pages_to_extract:
                pages_to_extract[doc_name].update(pages)
            else:
                pages_to_extract[doc_name] = pages
    return pages_to_extract


def load_chunks_from_json(chunk_file_path: Path) -> List[str]:
    """Load chunks from a JSON file."""
    try:
        with open(chunk_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('chunks', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load chunks from {chunk_file_path}: {e}")
        return []


def find_chunks_for_document_page(parser: str, chunker: str, doc_name: str, page_num: int, overlap_size: int) -> List[str]:
    """Find chunks for a specific document and page from the chunk files."""
 
    chunk_dir = CHUNKS_BASE_DIR / parser / chunker / f"overlap_{overlap_size}"
    
    # TableQuest naming pattern: {doc_name}_p{page_num}_{parser}_chunks.json
    pattern = f"{doc_name}_p{page_num}_{parser}_chunks.json"
    chunk_file = chunk_dir / pattern
    
    if chunk_file.exists():
        return load_chunks_from_json(chunk_file)
    
    # If specific page file doesn't exist, try document-level file without page number
    pattern = f"{doc_name}_{parser}_chunks.json"
    chunk_file = chunk_dir / pattern
    
    if chunk_file.exists():
        all_chunks = load_chunks_from_json(chunk_file)
        # For document-level chunks, we can't easily filter by page, so return all
        return all_chunks
    
    return []


def create_chunked_dataset(parser: str, chunker: str, overlap_size: int, records: List[Dict]) -> List[Dict]:
    """Create a chunked dataset for a specific parser-chunker combination."""
    new_rows = []
    pages_to_extract = _collect_pages_to_extract(records)
    
    print(f"Processing {parser}-{chunker}-overlap_{overlap_size} combination...")
    
    for doc_record in tqdm(records, desc=f"Processing {parser}-{chunker}-overlap_{overlap_size}"):
        doc_name = doc_record["doc_name"]
        
        document_processed = False
        for qa in doc_record["qa_pairs"]:
            question = qa["question"]
            answer = qa["answer"]
            question_processed = False
            
            for page_num in qa["evidence_pages"]:
                if page_num not in pages_to_extract.get(doc_name, ()):
                    continue
                
                # Get chunks for this document and page
                chunks = find_chunks_for_document_page(parser, chunker, doc_name, page_num, overlap_size)
                
                if not chunks:
                    tqdm.write(f"Warning: no chunks found for {doc_name} page {page_num} with {parser}-{chunker}-overlap_{overlap_size}")
                    continue
                
                for idx, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():
                        continue
                    
                    chunk_id = f"{doc_name}_page_{page_num}_{idx}_{parser}_{chunker}_overlap_{overlap_size}"
                    new_rows.append({
                        "query": question,
                        "document": doc_name,
                        "answer": answer,
                        "page_number": page_num,
                        "chunk_type": "text",
                        "text_description": chunk_text.strip(),
                        "chunk_id": chunk_id,
                        "image_filename": f"{doc_name}_{page_num}",
                    })
                    question_processed = True
                    document_processed = True
            
            # Ensure the question is included even if no chunks were generated
            if not question_processed:
                new_rows.append({
                    "query": question,
                    "document": doc_name,
                    "answer": answer,
                    "page_number": None,
                    "chunk_type": "text",
                    "text_description": "",
                    "chunk_id": f"{doc_name}_no_chunks_{parser}_{chunker}_overlap_{overlap_size}",
                    "image_filename": "",
                })
                document_processed = True
        
        # Ensure the document is included even if no valid pages were found
        if not document_processed:
            new_rows.append({
                "query": "",
                "document": doc_name,
                "answer": "",
                "page_number": None,
                "chunk_type": "text",
                "text_description": "",
                "chunk_id": f"{doc_name}_no_pages_{parser}_{chunker}_overlap_{overlap_size}",
                "image_filename": "",
            })
    
    return new_rows


def main() -> None:
    # Read JSON annotations
    with open(JSON_PATH, encoding="utf-8") as fh:
        records = json.load(fh)
    if LIMIT > 0:
        records = records[:LIMIT]
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each parser-chunker-overlap combination
    for overlap_size in OVERLAP_SIZES:
        print(f"\n{'*'*60}")
        print(f"Processing overlap size: {overlap_size}")
        print('*'*60)
        
        # Create overlap-specific output directory
        overlap_output_dir = OUTPUT_DIR / f"overlap_{overlap_size}"
        overlap_output_dir.mkdir(parents=True, exist_ok=True)
        
        for parser in PARSERS:
            for chunker in CHUNKERS:
                # Check if the parser-chunker combination exists
                chunk_dir = CHUNKS_BASE_DIR / parser / chunker / f"overlap_{overlap_size}"
                if not chunk_dir.exists():
                    print(f"Skipping {parser}-{chunker}-overlap_{overlap_size}: directory not found at {chunk_dir}")
                    continue
                
                # Create chunked dataset
                rows = create_chunked_dataset(parser, chunker, overlap_size, records)
                
                if not rows:
                    print(f"⚠️  No rows generated for {parser}-{chunker}-overlap_{overlap_size}")
                    continue
                
                # Save to CSV in overlap-specific subdirectory
                df = pd.DataFrame(rows, columns=COLUMNS)
                output_file = overlap_output_dir / f"{parser}_{chunker}_chunked_pages.csv"
                df.to_csv(output_file, index=False)
                print(f"✅ Saved {len(df)} rows → {output_file}")
    
    print("🎉 All parser-chunker combinations processed!")


if __name__ == "__main__":
    main()
