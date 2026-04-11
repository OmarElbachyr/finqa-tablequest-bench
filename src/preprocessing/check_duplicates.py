import json
from collections import Counter

# Path to the JSON file
JSON_PATH = "datasets/document_qa_pairs.json"

def check_duplicates():
    """Check for duplicate questions in the JSON file."""
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    questions = []
    for record in data:
        for qa_pair in record.get("qa_pairs", []):
            question = qa_pair.get("question", "").lower()  # Convert question to lowercase
            questions.append(question)

    # Count occurrences of each question
    question_counts = Counter(questions)

    # Find duplicates
    duplicates = {q: count for q, count in question_counts.items() if count > 1}

    if duplicates:
        print("Duplicate questions found:")
        for question, count in duplicates.items():
            print(f"{question}: {count} times")
    else:
        print("No duplicate questions found.")

def count_docname_evidence_combinations():
    """Count how many unique (doc_name, evidence_page) combinations exist in the JSON file."""
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    combinations = []
    for record in data:
        doc_name = record.get("doc_name", "")
        for qa_pair in record.get("qa_pairs", []):
            for page in qa_pair.get("evidence_pages", []):
                combinations.append((doc_name, page))

    unique_combinations = set(combinations)
    print(f"Total unique (doc_name, evidence_page) combinations: {len(unique_combinations)}")

if __name__ == "__main__":
    check_duplicates()
    count_docname_evidence_combinations()
