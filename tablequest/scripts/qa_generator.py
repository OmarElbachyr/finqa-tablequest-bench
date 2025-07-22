import os, json, base64, time, re
from pathlib import Path
import openai
from openai import OpenAI
import csv
from dotenv import load_dotenv
from typing import List, Optional, Literal
from pydantic import BaseModel



class Question(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    answer: str
    table_loc: str
    cot: Optional[str] = None


load_dotenv()
client = OpenAI()

# ---------- prompt templates ----------
PROMPTS = {
    "easy": """You are a junior equity analyst preparing a fact sheet for retail investors.

You are looking at a PNG image extracted from the {report_type} report of {company} for the year {year}.

1. Locate exactly one table with a clear, unique answer to a single-cell lookup question.
2. Draft that question.
3. Answer with the cell’s content only.
4. Return a JSON with keys: question, answer, table_loc.

""",
    "medium": """You are a senior credit-risk officer designing practice questions for interns.

You are looking at a PNG image extracted from the {report_type} report of {company} for the year {year}.

1. Pick one table.
2. Pose a question requiring a computed number not shown in the table.
3. Provide step-by-step reasoning in "cot", then the numeric answer only in "answer".
4. Return JSON with keys: question, answer, cot, table_loc.

""",
    "hard": """You are a university professor of corporate finance creating exam questions.

You are looking at a PNG image extracted from the {report_type} report of {company} for the year {year}.

1. Identify two or more tables on this page.
2. Craft an analytical question combining them.
3. Put detailed reasoning in "cot", concise conclusion (≤3 sentences) in "answer".
4. Return JSON with keys: question, answer, cot, table_loc.

"""
}

def img_b64(fp: Path) -> str:
    return base64.b64encode(fp.read_bytes()).decode()

def make_messages(prompt_text: str, img_blob_b64):
    system_msg = {"role": "system", "content": "You are a helpful vision-enabled assistant."}
    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_blob_b64}"}}
        ],
    }
    return [system_msg, user_msg]

def call_vlm(img_blob_b64, prompt_text):
    messages = make_messages(prompt_text, img_blob_b64)
    try:
        resp = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",  # Fixed model name #  gpt-4o-2024-08-06
            messages=messages,
            temperature=0.05,
            # max_tokens=800,
            response_format=Question
        )
        return resp.choices[0].message.content.strip()
    except openai.APIError as e:
        print(f"OpenAI API error: {e}")
        time.sleep(2)  # Back off on API errors
        return ""

def parse_qa_json(raw: str):
    """
    Parse a single-QA JSON blob from the model and return:
    question, answer, cot, table_loc
    """
    try:
        data = json.loads(raw)

        # Accept either {"questions":[{...}]} or a flat {...}
        if isinstance(data, dict) and "questions" in data:
            item = data["questions"][0]
        else:
            item = data

        return (
            item.get("question", ""),
            item.get("answer", ""),
            item.get("cot", ""),
            item.get("table_loc", ""),
        )
    except json.JSONDecodeError:
        print("⚠️  Invalid JSON returned from model")
        return "", "", "", ""


def process_page(fp: Path, difficulty: str, company: str, year: str, report_type: str):
    """
    Generate one QA pair for the given page. Returns a one-element list so the
    caller can still `.extend()` it.
    """
    attempts = 0
    img_blob_b64 = img_b64(fp)

    while attempts < 5:
        prompt = PROMPTS[difficulty].format(
            report_type=report_type,
            company=company,
            year=year
        )

        raw = call_vlm(img_blob_b64, prompt) 
        if not raw:
            attempts += 1
            time.sleep(1)
            continue

        try:
            q, a, cot, table_loc = parse_qa_json(raw)
        except Exception as e:
            print(f"⚠️  Failed to parse output of {fp}: {e}")
            attempts += 1
            time.sleep(1)
            continue

        qa_item = {
            "image": str(fp),  # Store relative path
            "difficulty": difficulty,
            "question": q,
            "answer": a,
            "cot": cot,              # may be empty for “easy”
            "table_loc": table_loc,
        }
        return qa_item

def parse_filename(stem: str):
    """
    Expected stem pattern:
      <company>_<year>_<type>_p<page>
      e.g.  JOHNSON-JOHNSON_2022Q4_EARNINGS_p14
    """
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Un-recognised filename format: {stem}")
    page_num   = int(re.sub(r"^[pP]", "", parts[-1]))
    report_type = parts[-2]
    year        = parts[-3]
    company     = "_".join(parts[:-3])      # in case company itself has underscores
    return company, year, report_type, page_num


def run(root="sampled_pages", output="qa_output"):
    root = Path(root)
    out_root = Path(output)
    out_root.mkdir(exist_ok=True)

    for diff in ("easy", "medium", "hard"):
        diff_dir = root / diff
        out_dir  = out_root / diff
        out_dir.mkdir(exist_ok=True)

        files = sorted(diff_dir.glob("*.png"))
        total = len(files)
        all_answers = []

        for idx, fp in enumerate(files, start=1):
            print(f"Processing {idx}/{total} — {fp.name} ({diff})")
            try:
                company, year, rtype, page = parse_filename(fp.stem)
                answer_json = process_page(
                    fp, diff, company, year, out_dir  # adjust signature if process_page needs rtype/page
                )
                answer_json.update(            # keep the extra metadata if useful
                    {"report_type": rtype, "page_number": page}
                )
                all_answers.append(answer_json)
            except Exception as e:
                print(f"⚠️  {fp.name}: {e}")

        with (out_root / f"{diff}.json").open("w") as f:
            json.dump(all_answers, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    root = 'tablequest/sampled_pages'
    output = 'tablequest/qa_pairs'
    run(root, output)