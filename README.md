# An Empirical Study of Retrieval‑Augmented Generation for Question Answering

This repository contains the code, data, and experiments for our systematic study of how **PDF parsing** and **text chunking** choices affect end‑to‑end Retrieval‑Augmented Generation (RAG) performance on financial documents. It also hosts **TableQuest**, a new table‑focused QA benchmark built from real-world SEC filings and earnings reports.

---

**Overview**  

This work presents the first systematic investigation of how key design choices in a Retrieval‑Augmented Generation (RAG) pipeline -PDF parsing and text chunking- impact end‑to‑end QA performance over financial documents, and provide actionable guidance for practitioners.

---

**Key Contributions**  

- **QA‑Centered Evaluation:** We frame PDF understanding as a question‑answering task to mirror real analytical workflows.  
- **Benchmarks:** We leverage two financial QA datasets, including **TableQuest**, our newly released table‑focused benchmark.  
- **Component Analysis:** We compare multiple open‑source PDF parsers and six common chunking strategies (with varied overlap), and study their interactions.  
- **Practical Guidelines:** Our results offer clear recommendations for building robust RAG systems on PDF corpora.

<!-- **Code & Data**  
All code, datasets (including TableQuest), and experiment configurations are publicly available on GitHub.   -->

---

## TableQuest: A Table‑Focused Financial QA Benchmark
**TableQuest** targets QA over tables in financial documents (10‑K, 10‑Q, earnings reports). Each sample includes the PDF page(s) with tables plus QA pairs, organized by difficulty.

Source PDFs originate from the [FinanceBench dataset](https://github.com/patronus-ai/financebench).

### Difficulty Levels
The dataset is organized into three difficulty tiers based on the cognitive complexity required to answer the questions:

| Tier | Definition | Cognitive steps required | Example |
|------|------------|--------------------------|---------|
| **Easy** (single-table extractive) | Answer is copied verbatim from **one table**—a single cell, header, or total. | 1. Locate the table.<br>2. Read the target cell. | What is the total amount of future maturities of long-term debt for 2026? |
| **Medium** (single-table numerical) | Answer requires ≤ 2 arithmetic operations (add, subtract, ratio, % change, etc.) **within one table**. | 1. Find the table.<br>2. Identify 2–3 cells.<br>3. Compute result. | What is the total of "Accruals not currently deductible" and "Pension costs" for the year 2022? |
| **Hard** (multi-table cross-table) | Answer requires combining data from **≥ 2 tables on the same page**, often plus a small calculation or comparison. | 1. Detect all tables.<br>2. Select related tables.<br>3. Align rows/columns.<br>4. Merge or compare values (and optionally compute). | Analyze the impact of special items on the operating income margin for both the "Safety and Industrial" and "Transportation and Electronics" segments. How do these adjustments affect the overall financial performance of 3M in Q2 2023? |

---

## Directory Structure

```
├── tablequest/                    # TableQuest dataset
│   ├── metadata/                  # Page and table metadata, sampling information
│   ├── prompts/                   # LLM prompt templates for different difficulty levels
│   ├── qa_pairs/                  # JSON files with question-answer pairs
│   ├── sampled_pages_pdf/         # Sampled pages organized by difficulty (PDF files)
│   ├── scripts/                   # Data processing and QA generation scripts
│   └── stats/                     # Statistics and analysis scripts/notebooks
├── new_scripts/                   # Evaluation framework
│   ├── chunkers/                  # Text chunking implementations
│   ├── data/                      # Generated answers and intermediate files (parsing, chunking results... etc)
│   ├── evaluation/                # Evaluation metrics and classes (retrieval and answer correctness)
│   ├── generators/                # Answer generation modules
│   ├── parsers/                   # PDF parsing implementations
│   ├── preprocessing/             # Data preprocessing utilities
│   ├── retrievers/                # Information retrieval methods (BM25, dense, hybrid, ColBert)
│   ├── stats/                     # Stats utilities
│   └── test/                      # End-to-end testing and evaluation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd finqa-tablequest-bench

# Create and activate Conda env
conda create -n finqa-tablequest python=3.11 -y
conda activate finqa-tablequest

# Install dependencies
pip install -r requirements.txt
```

Note: If you encounter issues with installation, you can comment out the 'Layout detection dependencies' section in requirements.txt, install the remaining packages, then uncomment and reinstall.

## Usage

### Core Pipeline Components

#### 1. PDF Parsing
```bash
# Parse PDF documents into text
python new_scripts/parsers/text_parsers.py
```

#### 2. Text Chunking
```bash
# Chunk parsed text using different strategies
python new_scripts/chunkers/chunkers.py
```

#### 3. Information Retrieval & Evaluation
```bash
# End-to-end indexing, retrieval & evaluation pipeline
python new_scripts/test/retrieval_evaluation.py
```

#### 4. Answer Generation
```bash
# Generate answers using Ollama models
python new_scripts/generators/ollama_answer_generation.py
```

#### 5. Evaluation
```bash
# Answer correctness evaluation
python new_scripts/evaluation/llm_judge_evaluation.py
```

### Supported Tools

- **Parsers**: PyPDF2, PyMuPDF, pdfplumber, pypdfium2, docling, unstructured
- **Chunkers**: token, sentence, semantic, recursive, SDPM, neural
- **Retrievers**: BM25, dense embeddings, ColBERT, SPLADE, ColPali