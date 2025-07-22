import json
from pathlib import Path
import fitz

INPUT = Path("new_scripts/data/csv/document_qa_pairs.json")
PDF_DIR = Path("financebench_pdfs")
OUTPUT_DIR = Path("new_scripts/data/financebench_extracted_pages_pdf")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)
    
for rec in data:
    doc_name = rec["doc_name"]
    pdf_path = PDF_DIR / f"{doc_name}.pdf"
    if not pdf_path.exists():
        continue
    
    # Collect all unique evidence pages from all qa_pairs
    pages = set()
    for qa_pair in rec.get("qa_pairs", []):
        pages.update(qa_pair.get("evidence_pages", []))
    
    if not pages:
        continue
    
    doc = fitz.open(pdf_path)
    for page_num in pages:
        if 1 <= page_num <= doc.page_count:
            out_pdf = fitz.open()
            out_pdf.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
            out_path = OUTPUT_DIR / f"{doc_name}_{page_num}.pdf"
            out_pdf.save(out_path)
            out_pdf.close()
    doc.close()
