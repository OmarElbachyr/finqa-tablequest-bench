# LLM Voting Verification for TableQuest QA Pairs

Automatically verify TableQuest QA pairs using OpenAI vision models with majority voting across multiple models.

**Models used:** `gpt-5-2025-08-07`, `o3-2025-04-16`, `gpt-4.1-2025-04-14`

## Overview

The `openai_voter.py` script evaluates QA pairs by:
1. Showing the financial document image to multiple OpenAI models
2. Asking each model to judge if the answer is CORRECT or INCORRECT
3. Using majority voting to determine the final judgment

## Setup

### Prerequisites

1. **OpenAI API key** in `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   ```

2. **Python dependencies**:
   ```bash
   pip install openai python-dotenv
   ```

## Usage

Edit the script configuration at the bottom:

```python
DIFFICULTY = "hard"          # easy, medium, or hard
MODELS = ["gpt-5-2025-08-07", "o3-2025-04-16", "gpt-4.1-2025-04-14"]  # OpenAI models
LIMIT = None                 # None for all items, or specific number for testing
```

Then run:

```bash
python3 openai_voter.py
```

## Output

Results are saved to: `tablequest_dataset/annotation/verification_openai/verification_{DIFFICULTY}_{TIMESTAMP}.json`

Each result includes:
- **item**: Original QA pair with image, question, and answer
- **verifications**: Individual model judgments and responses
- **majority_vote**: Final verdict (CORRECT or INCORRECT)

## Post-Voting Processing

After running the voting verification, you can analyze results and enrich your data:

### Analysis Scripts

#### Quick Statistics
```bash
python3 analyze_vote_results.py
```
- Shows per-difficulty summary statistics:
  - Majority vote counts (CORRECT vs INCORRECT)
  - Per-model performance breakdown
  - Voting pattern distribution (e.g., 3C-0I, 2C-1I, etc.)
