from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple, Set, Dict

import pandas as pd
from tqdm import tqdm
from pdfminer.high_level import extract_text

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
JSON_PATH = Path("tablequest/qa_pairs/adapted_pairs/adapted_hard.json")
PDF_DIR = Path("financebench_pdfs")
CSV_PATH = Path("new_scripts/data/chunks/test/tq_chunked_pages_test.csv")
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
            pages_to_extract[doc_name] = pages
    return pages_to_extract


# ---------------------------------------------------------------------------
# PDF parsing & chunking functions
# ---------------------------------------------------------------------------

def parse_pdf_page(pdf_path: Path, page_num: int) -> str:
    """
    Extract raw text from a single page of a PDF using pdfminer.
    
    Args:
        pdf_path: path to the PDF file
        page_num: 1-indexed page number to extract
    
    Returns:
        The plain text of that page.
    """
    # pdfminer expects 0-indexed page_numbers
    return extract_text(str(pdf_path), page_numbers=[page_num - 1])


def chunk_text_simple(text: str, max_chars: int = 1000) -> List[str]:
    """
    Split text into simple, fixed-size character chunks.
    
    Args:
        text: the full text to chunk
        max_chars: maximum number of characters per chunk
    
    Returns:
        A list of text chunks.
    """
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _extract_page_chunks(
    pdf_path: Path, page_num: int, doc_name: str
) -> List[Tuple[str, str, str]]:
    """
    Parse a PDF page and produce (chunk_text, "text", chunk_id) tuples
    using pdfminer + simple character-based chunking.
    """
    raw = parse_pdf_page(pdf_path, page_num)
    chunks = chunk_text_simple(raw)
    results: List[Tuple[str, str, str]] = []
    for idx, chunk in enumerate(chunks):
        text = chunk.strip()
        if not text:
            continue
        chunk_id = f"{doc_name}_page_{page_num}_{idx}"
        results.append((text, "text", chunk_id))
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    # Read JSON annotations
    with open(JSON_PATH, encoding="utf-8") as fh:
        records = json.load(fh)
    if LIMIT > 0:
        records = records[:LIMIT]

    # Which pages to pull
    pages_to_extract = _collect_pages_to_extract(records)

    new_rows = []
    # Process each document & QA
    for doc_record in tqdm(records, desc="Processing documents"):
        doc_name = doc_record["doc_name"]
        pdf_path = PDF_DIR / f"{doc_name}.pdf"
        if not pdf_path.exists():
            tqdm.write(f"Warning: PDF not found: {pdf_path}")
            continue

        document_processed = False  # Track if the document is processed
        for qa in doc_record["qa_pairs"]:
            question = qa["question"]
            answer = qa["answer"]
            question_processed = False  # Track if the question is processed

            for page_num in qa["evidence_pages"]:
                if page_num not in pages_to_extract.get(doc_name, ()): 
                    continue

                # Parse & chunk via pdfminer + simple splitter
                page_chunks = _extract_page_chunks(pdf_path, page_num, doc_name)
                if not page_chunks:
                    tqdm.write(f"Warning: no chunks from {doc_name} page {page_num}")
                    continue

                for chunk_text, chunk_type, chunk_id in page_chunks:
                    new_rows.append({
                        "query": question,
                        "document": doc_name,
                        "answer": answer,
                        "page_number": page_num,
                        "chunk_type": chunk_type,
                        "text_description": chunk_text,
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
                    "chunk_id": f"{doc_name}_no_chunks",
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
                "chunk_id": f"{doc_name}_no_pages",
                "image_filename": "",
            })

    # Write out
    if not new_rows:
        print("✅ No rows to save.")
        return

    df_new = pd.DataFrame(new_rows, columns=COLUMNS)
    df_new.to_csv(CSV_PATH, index=False)
    print(f"✅ Saved {len(df_new)} rows → {CSV_PATH}")


if __name__ == "__main__":
    main()
