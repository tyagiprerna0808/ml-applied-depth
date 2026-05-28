"""
search_corpus.py — search a real corpus by MEANING, not keywords.

We load 5,000 short news passages, turn each into a 384-number vector,
then for any English query we ask: "which 5 passages are closest in meaning?"
We also measure how long the search takes.

The point: this works with ZERO keyword matching. Query "tennis player wins
grand slam" will find tennis news even if those exact words don't appear,
because the embeddings capture meaning, not surface text.
"""

import os
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# We cache embeddings to a .npy file so re-runs are instant.
os.makedirs("data", exist_ok=True)
EMB_CACHE = "data/agnews_embeddings.npy"
TXT_CACHE = "data/agnews_texts.npy"
N = 5000  # how many passages to load


# ---------- Step 1: load 5,000 news passages ----------
if not os.path.exists(TXT_CACHE):
    print("Downloading ag_news dataset (first time, ~30s)...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    texts = [ds[i]["text"] for i in range(N)]
    np.save(TXT_CACHE, np.array(texts, dtype=object))
else:
    print("Loading cached texts...")
    texts = np.load(TXT_CACHE, allow_pickle=True).tolist()

print(f"Corpus size: {len(texts):,} passages")
print(f"Example passage: {texts[0][:150]}...\n")


# ---------- Step 2: embed all 5,000 passages ----------
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

if not os.path.exists(EMB_CACHE):
    print(f"Embedding {N:,} passages (~2-3 min on CPU)...")
    t0 = time.time()
    corpus_vectors = model.encode(texts, batch_size=64, show_progress_bar=True)
    print(f"Done in {time.time()-t0:.1f}s.")
    np.save(EMB_CACHE, corpus_vectors)
else:
    print("Loading cached embeddings...")
    corpus_vectors = np.load(EMB_CACHE)

print(f"Corpus matrix shape: {corpus_vectors.shape}  (5000 passages x 384 numbers each)\n")

# Pre-normalize each vector to length 1. Then cosine similarity = simple dot product.
# Faster, and one of the standard ANN tricks we'll see again.
corpus_norm = corpus_vectors / np.linalg.norm(corpus_vectors, axis=1, keepdims=True)


# ---------- Step 3: brute-force top-k search ----------
def find_top_k(query_vec_normalized, k=5):
    """Find the k passages most similar to the query vector.
    'Brute force' because we compare against every single passage."""
    scores = corpus_norm @ query_vec_normalized          # one matrix-vector multiply
    top_k_idx = np.argpartition(-scores, k)[:k]           # k highest scores (unordered)
    top_k_idx = top_k_idx[np.argsort(-scores[top_k_idx])] # sort those k
    return [(float(scores[i]), texts[i]) for i in top_k_idx]


# ---------- Step 4: try a few queries and time them ----------
queries = [
    "tennis player wins grand slam",
    "earnings report from a tech company",
    "conflict in the middle east",
    "scientists discover something new in space",
]

for q in queries:
    # Embed the query (small, ~50ms — not part of search timing)
    qv = model.encode([q])[0]
    qv = qv / np.linalg.norm(qv)

    # Time JUST the search step (not the query embedding)
    t0 = time.perf_counter()
    results = find_top_k(qv, k=3)
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"\nQUERY: {q}")
    print("-" * 70)
    for score, text in results:
        print(f"  [{score:.3f}]  {text[:130]}...")
    print(f"  search latency: {latency_ms:.1f} ms")
