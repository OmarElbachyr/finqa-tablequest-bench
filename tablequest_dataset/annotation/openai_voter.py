#!/usr/bin/env python3
"""
Simplified LLM Voting Verification Script for TableQuest QA Pairs (OpenAI version)

This script uses multiple OpenAI models to verify the correctness of QA pairs
by showing them the image and generated question/answer. Uses majority voting
across models.
"""

import json
import os
import sys
import asyncio
import base64
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OpenAIVotingVerifier:
    def __init__(self, models: List[str] = None):
        """
        Initialize the verifier.

        Args:
            models: List of OpenAI model names
        """
        self.models = models or ["gpt-4o-mini", "gpt-4o"]

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found. Please add it to your .env file.")
            sys.exit(1)

        self.client = OpenAI(api_key=api_key)

    def _encode_image(self, image_path: Path) -> str:
        """Encode image as base64 string for inline usage."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None

    def _create_verification_prompt(self, qa_item: Dict[str, Any], difficulty: str) -> str:
        """Create a prompt for LLM verification."""
        prompt_path = Path("tablequest_dataset/annotation/prompts/voting_prompt.txt")
        with open(prompt_path, "r") as f:
            template = f.read()

        prompt = template.format(
            report_type=qa_item.get('report_type', 'Unknown'),
            page_number=qa_item.get('page_number', 'Unknown'),
            table_loc=qa_item.get('table_loc', 'Unknown'),
            question=qa_item['question'],
            answer=qa_item['answer'],
            cot=(qa_item.get('cot', '') if difficulty in ["medium", "hard"] and qa_item.get("cot") else '')
        )
        return prompt

    async def _verify_with_model(
        self, model: str, image_base64: str, prompt: str
    ) -> Dict[str, Any]:
        """Verify QA pair with one model."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        ],
                    }
                ],
            )

            content = response.choices[0].message.content.strip().upper()

            if "INCORRECT" in content:
                judgment = "INCORRECT"
            else:
                judgment = "CORRECT"

            return {"model": model, "judgment": judgment, "response": content}

        except Exception as e:
            return {"model": model, "judgment": "INCORRECT", "response": f"Error: {str(e)}"}

    async def verify_qa_item(self, qa_item: Dict[str, Any], difficulty: str) -> Dict[str, Any]:
        """Verify one QA item with all models."""
        image_path = Path(qa_item["image"])

        if not image_path.exists():
            return {
                "item": qa_item,
                "verifications": [],
                "majority_vote": "INCORRECT",
            }

        image_base64 = self._encode_image(image_path)
        if not image_base64:
            return {
                "item": qa_item,
                "verifications": [],
                "majority_vote": "INCORRECT",
            }

        prompt = self._create_verification_prompt(qa_item, difficulty)

        verifications = []
        for model in self.models:
            verifications.append(await self._verify_with_model(model, image_base64, prompt))

        correct_votes = sum(1 for v in verifications if v["judgment"] == "CORRECT")
        incorrect_votes = sum(1 for v in verifications if v["judgment"] == "INCORRECT")

        majority_vote = "CORRECT" if correct_votes >= incorrect_votes else "INCORRECT"

        return {
            "item": qa_item,
            "verifications": verifications,
            "majority_vote": majority_vote,
        }

    async def verify_dataset(self, difficulty: str, max_items: int = None) -> Dict[str, Any]:
        """Verify all QA pairs for a given difficulty."""
        qa_file = Path("tablequest_dataset") / "qa_pairs" / f"{difficulty}.json"
        with open(qa_file, "r") as f:
            qa_data = json.load(f)

        if max_items:
            qa_data = qa_data[:max_items]

        results = []
        for idx, qa_item in enumerate(qa_data, 1):
            logger.info(f"Processing {idx}/{len(qa_data)}")
            results.append(await self.verify_qa_item(qa_item, difficulty))

        return {
            "metadata": {
                "difficulty": difficulty,
                "timestamp": datetime.now().isoformat(),
                "models_used": self.models,
                "total_items": len(results),
            },
            "all_results": results,
        }

    def save_results(self, results: Dict[str, Any], output_file: str):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")


async def main():
    DIFFICULTY = "hard"
    MODELS = ["gpt-5-2025-08-07",
              "o3-2025-04-16",
              "gpt-4.1-2025-04-14"]
    # MODELS = ["gpt-4.1-2025-04-14"]

    # LIMIT controls how many QA items to verify. Set to None to verify all items.
    LIMIT = None  # e.g. LIMIT = 50

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = Path("tablequest_dataset") / "annotation" / "verification_openai" / f"verification_{DIFFICULTY}_{timestamp}.json"

    verifier = OpenAIVotingVerifier(models=MODELS)
    results = await verifier.verify_dataset(DIFFICULTY, max_items=LIMIT)

    verifier.save_results(results, OUTPUT_FILE)

    print("\nVerification completed.")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
