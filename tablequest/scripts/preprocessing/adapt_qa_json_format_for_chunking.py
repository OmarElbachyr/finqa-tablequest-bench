import json


difficulty = "hard"
# Load medium.json
with open(f'tablequest/qa_pairs/{difficulty}.json', 'r') as f:
    medium_data = json.load(f)

# Adapt medium.json to the structure of document_qa_pairs.json
adapted_data = []
for item in medium_data:
    adapted_item = {
        # "doc_name": item["image"].split("_")[0],  # Extract doc_name from image
        "doc_name": item["image"].split("/")[-1].rsplit("_p", 1)[0],  # Extract doc_name from image
        "qa_pairs": [
            {
                "question": item["question"],
                "answer": item["answer"],
                "evidence_pages": [item["page_number"]],  # Wrap page_number in a list
                "difficulty": item["difficulty"],
                "cot": item["cot"],
                "table_loc": item["table_loc"],
                "report_type": item["report_type"]
            }
        ]
    }
    adapted_data.append(adapted_item)

# Save the adapted data to a new JSON file
out_file = f'tablequest/qa_pairs/adapted_pairs/adapted_{difficulty}.json'
with open(out_file, 'w') as f:
    json.dump(adapted_data, f, indent=4)
print(f"Adapted data saved to {out_file}")