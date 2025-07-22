import os
import csv
import fitz  # PyMuPDF
import layoutparser as lp
import numpy as np
from PIL import Image, ImageFont

# ─── Pillow compatibility (for old Pillow versions) ─────────────────────
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.Resampling.BILINEAR

if not hasattr(ImageFont.FreeTypeFont, "getsize"):
    def _getsize(font, text):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    ImageFont.FreeTypeFont.getsize = _getsize

# ─── Configuration ──────────────────────────────────────────────────────
SAMPLED_PDF_DIR   = "/home/omar/datasets/financebench_pdfs"
COUNTS_CSV_PATH   = "tablequest/metadata/table_counts_by_doc_check_types.csv"  # <-- NEW
TABLE_LABELS      = {"Table"}  # block types we consider “tables”

# Layout-detection model (PubLayNet Mask-RCNN)
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
)

os.makedirs(os.path.dirname(COUNTS_CSV_PATH), exist_ok=True)

# ─── Helper to parse filename → company, year, type ─────────────────────
def parse_pdf_filename(pdf_filename_base: str) -> tuple[str, str, str]:
    types = {"10K", "10Q", "8K", "EARNINGS", "10K_ANNUAL"}
    parts = pdf_filename_base.split("_")
    default = ("UnknownCompany", "UnknownYear", "UnknownType")

    if len(parts) >= 3:
        company = "_".join(parts[:-2])
        year = parts[-2]
        type_candidate = parts[-1].upper()
        report_type = next((t for t in types if t in type_candidate), "UnknownType")
        return company, year, report_type

    if len(parts) == 2:
        p0, p1 = parts
        if p1.isdigit() and len(p1) == 4:
            return p0, p1, "UnknownType"
        if p0.isdigit() and len(p0) == 4:
            return "UnknownCompany", p0, p1
        return p0, "UnknownYear", p1

    if len(parts) == 1 and parts[0]:
        return parts[0], "UnknownYear", "UnknownType"

    return default


# ─── Main loop: count tables per PDF ────────────────────────────────────
doc_metadata = []
for filename in sorted(os.listdir(SAMPLED_PDF_DIR)):
    if not filename.lower().endswith(".pdf"):
        continue
    pdf_path  = os.path.join(SAMPLED_PDF_DIR, filename)
    base_name = os.path.splitext(filename)[0]
    company_name, report_year, report_type = parse_pdf_filename(base_name)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌  Could not open {filename}: {e}")
        continue

    table_count = 0

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        pix  = page.get_pixmap(dpi=200)
        mode = "RGBA" if pix.alpha else "RGB"
        img  = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        if mode == "RGBA":
            img = img.convert("RGB")

        layout = model.detect(np.array(img))
        table_count += sum(1 for block in layout if block.type in TABLE_LABELS)

    doc_metadata.append(
        {
            "document_name": base_name,
            "company_name": company_name,
            "report_type": report_type,
            "report_year": report_year,
            "table_count": table_count,
        }
    )
    print(f"→ {filename}: {table_count} tables")

# ─── Write CSV ──────────────────────────────────────────────────────────
if doc_metadata:
    fieldnames = [
        "document_name",
        "company_name",
        "report_type",
        "report_year",
        "table_count",
    ]
    with open(COUNTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(doc_metadata)
    print(f"✅  Saved table counts for {len(doc_metadata)} PDFs → {COUNTS_CSV_PATH}")
else:
    print("ℹ️  No PDFs processed – no CSV written.")