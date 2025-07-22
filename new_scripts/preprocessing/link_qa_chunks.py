import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ------------------
# User‑configured paths
# ------------------
QA_JSON_PATH    = 'datasets/document_qa_pairs.json'               # QA pairs
CHUNKS_DIR      = 'new_scripts/data/parsed_pages_chunks/pdfminer'  # directory with *_chunks.json files
OUTPUT_CSV_PATH = 'csv/link_qa_chunks.csv'                     # resulting CSV

# ------------------
# Helpers
# ------------------

def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv(rows: List[Dict[str, str]], path: str) -> None:
    if not rows:
        print('No rows to write — CSV not created.')
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ------------------
# Build an in‑memory index of chunks for quick lookup
# ------------------
CHUNK_FILENAME_RE = re.compile(r'^(?P<doc>.+?)_page_(?P<page>\d+).*_chunks\.json$')


def build_chunk_index(chunks_root: str) -> Dict[Tuple[str, int], List[str]]:
    """Scan *once* and map (doc_name, page_number) → [chunks]."""
    index: Dict[Tuple[str, int], List[str]] = {}
    for path in Path(chunks_root).rglob('*_chunks.json'):
        match = CHUNK_FILENAME_RE.match(path.name)
        if not match:
            continue  # skip unrecognised files

        doc_name   = match.group('doc')
        page_number = int(match.group('page'))
        data = load_json(path)
        chunks = data.get('chunks') if isinstance(data, dict) else data  # accept either shape
        if isinstance(chunks, list):
            index[(doc_name, page_number)] = chunks
    return index


CHUNK_INDEX = build_chunk_index(CHUNKS_DIR)
print(f"Indexed {len(CHUNK_INDEX)} (doc, page) pairs from {CHUNKS_DIR}")
print('Total chunk count:', sum(len(v) for v in CHUNK_INDEX.values()))

empty_chunks_count = sum(chunk == "" for chunks in CHUNK_INDEX.values() for chunk in chunks)
print("Empty chunks found:", empty_chunks_count)


# ------------------
# Link QA pairs to chunks
# ------------------

def link_qa_to_chunks(qa_json_path: str) -> List[Dict[str, str]]:
    """Return one CSV row per chunk on every evidence page."""
    rows: List[Dict[str, str]] = []
    qa_data = load_json(qa_json_path)

    for doc in qa_data:
        doc_name = doc['doc_name']
        for qa in doc['qa_pairs']:
            question = qa['question']
            answer   = qa['answer']
            for page_num in qa['evidence_pages']:
                chunks = CHUNK_INDEX.get((doc_name, page_num), []) or ['']
                for i, chunk in enumerate(chunks):
                    rows.append({
                        'doc_name':    doc_name,
                        'question':    question,
                        'answer':      answer,
                        'page_number': page_num,
                        'chunk':       chunk,
                        'chunk_id': f"{doc_name}_{page_num}_{i}",
                        'page_id': f"{doc_name}_{page_num}"
                    })
    return rows


# ------------------
# Run pipeline
# ------------------
all_rows = link_qa_to_chunks(QA_JSON_PATH)
save_csv(all_rows, OUTPUT_CSV_PATH)
print(f"CSV written to {OUTPUT_CSV_PATH}  |  rows: {len(all_rows)}")
