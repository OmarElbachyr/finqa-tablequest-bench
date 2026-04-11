import json
from collections import Counter, defaultdict

def compute_stats(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    results = data["all_results"]
    models = data["metadata"]["models_used"]

    total = len(results)
    majority_counts = Counter(r["majority_vote"] for r in results)

    # Per-model stats
    model_stats = defaultdict(Counter)
    for r in results:
        for v in r["verifications"]:
            model_stats[v["model"]][v["judgment"]] += 1

    # Voting patterns
    pattern_counts = Counter()
    for r in results:
        votes = [v["judgment"] for v in r["verifications"]]
        c = votes.count("CORRECT")
        i = votes.count("INCORRECT")
        pattern_counts[f"{c}C-{i}I"] += 1

    # Print summary
    print("===== VERIFICATION STATISTICS =====")
    print(f"Total QA pairs: {total}")
    print(f"Majority Correct:   {majority_counts['CORRECT']} ({majority_counts['CORRECT']/total*100:.1f}%)")
    print(f"Majority Incorrect: {majority_counts['INCORRECT']} ({majority_counts['INCORRECT']/total*100:.1f}%)")
    print()

    print("----- Per Model -----")
    for m in models:
        correct = model_stats[m]["CORRECT"]
        incorrect = model_stats[m]["INCORRECT"]
        print(f"{m}: Correct {correct} ({correct/total*100:.1f}%), "
              f"Incorrect {incorrect} ({incorrect/total*100:.1f}%)")
    print()

    print("----- Voting Patterns -----")
    for pattern, count in pattern_counts.items():
        print(f"{pattern}: {count} ({count/total*100:.1f}%)")

if __name__ == "__main__":
    for diffuculty in ["easy", "medium", "hard"]:
        print(f"=== Difficulty: {diffuculty} ===")
        compute_stats(f"tablequest/annotation/verification_openai/verification_{diffuculty}.json")
        print("      ============================\n")
