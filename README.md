# ml-applied-depth

From-scratch Python implementations and experiments accompanying my blog series on applied AI/ML engineering depth at https://tyagiprerna0808.github.io.

The blog explains the concept; this repo is where each concept is implemented from scratch and the numbers are produced. Each post has:

- `notebook.ipynb` — the worked implementation
- `assignment.ipynb` — the same notebook with scaffolding, TODOs, and tests so you can implement it yourself before reading the worked version
- `experiments/` — scripts that produce the numbers and plots in the post
- `results/` — outputs (CSVs, plots)

## Series

1. Embeddings & vector search internals — HNSW / IVF / PQ from scratch
2. Retrieval: sparse, dense, hybrid — BM25 from scratch, cross-encoder reranking
3. Chunking & document representation
4. Evals for generative systems
5. RAG failure modes
6. Agent loops & tool use — ReAct from scratch
7. Context engineering & prompt caching
8. Fine-tuning at small scale — LoRA / QLoRA

## Setup

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
