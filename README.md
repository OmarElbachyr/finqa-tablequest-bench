# An Empirical Study of Retrieval‑Augmented Generation for Question Answering

**Overview**  
This work presents the first systematic investigation of how key design choices in a Retrieval‑Augmented Generation (RAG) pipeline—PDF parsing and text chunking—impact end‑to‑end QA performance over financial documents.

**Key Contributions**  
- **QA‑Centered Evaluation:** We frame PDF understanding as a question‑answering task to mirror real analytical workflows.  
- **Benchmarks:** We leverage two financial QA datasets, including **TableQuest**, our newly released table‑focused benchmark.  
- **Component Analysis:** We compare multiple open‑source PDF parsers and six common chunking strategies (with varied overlap), and explore their interactions.  
- **Practical Guidelines:** Our results offer clear recommendations for building robust RAG systems on PDF corpora.

**Code & Data**  
All code, datasets (including TableQuest), and experiment configurations are publicly available on GitHub.  
