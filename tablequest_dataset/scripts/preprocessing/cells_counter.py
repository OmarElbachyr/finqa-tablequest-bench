"""
Enrich sampled_pages_metadata.csv by
  1. extracting the sampled page into a 1-page PDF,
  2. parsing it with Docling,
  3. exporting every table on that page to Markdown,
  4. counting cells (rows × cols),
  5. saving markdown + min_cells.

Dependencies
------------
pip install docling pymupdf pandas pillow
"""

from pathlib import Path
import fitz                      # PyMuPDF
import pandas as pd
import tempfile, os
from docling.document_converter import DocumentConverter

# ── paths ──────────────────────────────────────────────────────────────
META_CSV      = Path("tablequest/metadata/tablequest_easy_medium.csv")
OUT_CSV       = Path("tablequest/metadata/tablequest_easy_medium_enriched.csv")

df = pd.read_csv(META_CSV)
df["tables_markdown"] = ""
df["min_cells"]       = None

def extract_page_to_temp(pdf_path: Path, page_no: int) -> Path:
    """Return a temp-file path containing only page_no (1-based) of pdf_path."""
    doc = fitz.open(pdf_path)
    single = fitz.open()                # empty PDF
    single.insert_pdf(doc, from_page=page_no-1, to_page=page_no-1)
    tmp = Path(tempfile.mkstemp(suffix=".pdf")[1])
    single.save(tmp)
    single.close(); doc.close()
    return tmp

conv = DocumentConverter()              # reuse one converter instance

for idx, row in df.iterrows():
    doc_path = row["document_path"]   # CORNING_2022_10K
    page_no   = int(row["page_number"])

    # 1-page extraction
    one_page_pdf = extract_page_to_temp(doc_path, page_no)

    # Docling parse
    doc = conv.convert(one_page_pdf).document
    tables = doc.tables                # all belong to that single page

    md_chunks, cell_counts = [], []
    for t in tables:
        md_chunks.append(t.export_to_markdown())
        f = t.export_to_dataframe()
        cell_counts.append(f.shape[0] * f.shape[1])

    df.at[idx, "tables_markdown"] = "\n\n---\n\n".join(md_chunks)
    df.at[idx, "min_cells"]       = min(cell_counts) if cell_counts else None

    os.remove(one_page_pdf)            # clean up temp file

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"✅  Saved enriched metadata → {OUT_CSV}")