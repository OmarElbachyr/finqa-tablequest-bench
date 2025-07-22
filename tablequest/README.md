# TableQuest

A table-focused benchmark dataset for question answering over financial documents.

## Overview

TableQuest is a dataset of table-based question-answer pairs extracted from financial documents (10-K, 10-Q, earnings reports). The dataset contains document pages with tables and corresponding questions with answers, organized by difficulty levels.

The original PDF financial reports used to create TableQuest are sourced from the [FinanceBench dataset](https://github.com/patronus-ai/financebench).

## Difficulty Levels

The dataset is organized into three difficulty tiers based on the cognitive complexity required to answer the questions:

| Tier | Definition | Cognitive steps required | Example |
|------|------------|--------------------------|---------|
| **Easy** (single-table extractive) | Answer is copied verbatim from **one table**—a single cell, header, or total. | 1. Locate the table.<br>2. Read the target cell. | What is the total amount of future maturities of long-term debt for 2026? |
| **Medium** (single-table numerical) | Answer requires ≤ 2 arithmetic operations (add, subtract, ratio, % change, etc.) **within one table**. | 1. Find the table.<br>2. Identify 2–3 cells.<br>3. Compute result. | What is the total of 'Accruals not currently deductible' and 'Pension costs' for the year 2022? |
| **Hard** (multi-table cross-table) | Answer requires combining data from **≥ 2 tables on the same page**, often plus a small calculation or comparison. | 1. Detect all tables.<br>2. Select related tables.<br>3. Align rows/columns.<br>4. Merge or compare values (and optionally compute). | Analyze the impact of special items on the operating income margin for both the 'Safety and Industrial' and 'Transportation and Electronics' segments. How do these adjustments affect the overall financial performance of 3M in Q2 2023? |

## Directory Structure

```
tablequest/
├── metadata/            # Page and table metadata, sampling information
├── prompts/             # LLM prompt templates for different difficulty levels: easy, medium and hard
├── qa_pairs/            # JSON files with question-answer pairs for different difficulty levels
├── sampled_pages_pdf/   # Sampled pages organized by difficulty (easy/medium/hard) saved as .pdf files
├── scripts/             # Data processing and QA generation scripts
└── stats/               # Statistics and analysis scripts/notebooks
```

## Components

- **metadata/**: Contains CSV files with page metadata, sampling information, and document mappings
- **prompts/**: LLM prompt templates organized by difficulty levels (easy, medium, hard) for question generation
- **qa_pairs/**: JSON files containing question-answer pairs for each difficulty level
- **sampled_pages_pdf/**: Document pages containing tables, organized by difficulty and saved as PDF files
- **scripts/**: Python scripts for data processing, PDF handling, table extraction, and QA generation
- **stats/**: Analysis scripts and Jupyter notebooks for dataset statistics and evaluation
