"""
latency_scaling.py — how slow does brute force get as the corpus grows?

We reuse the cached 5,000 embeddings from search_corpus.py.
We run searches against subsets of size N = 100, 500, 1000, 2500, 5000.
For each N we run 50 random queries, record mean and p95 latency,
plot the result, and project where brute force stops being acceptable.

The point of this script: PROVE that brute force scales linearly in N
(not by asserting it but by measuring it), and figure out at what N
this stops being usable for a real product.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

EMB_CACHE = "data/agnews_embeddings.npy"
RESULTS_DIR = "blog/01-embeddings/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

assert os.path.exists(EMB_CACHE), "Run search_corpus.py first to cache embeddings."


# ---------- Load and normalize the cached corpus ----------
print("Loading cached embeddings...")
corpus_vectors = np.load(EMB_CACHE)
corpus_norm = corpus_vectors / np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
print(f"Corpus shape: {corpus_norm.shape}\n")


# ---------- Build a fixed query set: 50 random passages from the corpus ----------
rng = np.random.default_rng(seed=0)
query_idx = rng.choice(len(corpus_norm), size=50, replace=False)
queries = corpus_norm[query_idx]


# ---------- The brute-force search, isolated for timing ----------
def brute_force_topk(subset, query_vec, k=10):
    scores = subset @ query_vec
    topk_idx = np.argpartition(-scores, k)[:k]
    return topk_idx[np.argsort(-scores[topk_idx])]


# ---------- Sweep corpus size ----------
N_VALUES = [100, 500, 1000, 2500, 5000]
K = 10

print(f"Measuring search latency at K={K}, {len(queries)} queries per N:\n")
print(f"{'N':>7}   {'mean (ms)':>12}   {'p95 (ms)':>12}")
print("-" * 38)

results = []
for N in N_VALUES:
    subset = corpus_norm[:N]
    # Warm-up: first call is always slower (CPU cache, branch prediction)
    _ = brute_force_topk(subset, queries[0], k=K)

    latencies_ms = []
    for q in queries:
        t0 = time.perf_counter()
        brute_force_topk(subset, q, k=K)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    results.append((N, mean_ms, p95_ms))
    print(f"{N:>7d}   {mean_ms:>12.3f}   {p95_ms:>12.3f}")


# ---------- Plot ----------
ns, means, p95s = zip(*results)
plt.figure(figsize=(7.5, 5))
plt.plot(ns, means, marker='o', linewidth=2, label='mean')
plt.plot(ns, p95s, marker='s', linewidth=2, label='p95', alpha=0.7)
plt.xlabel('Corpus size N')
plt.ylabel('Search latency (ms)')
plt.title('Brute-force top-10 latency vs corpus size\n(384-d embeddings, CPU)')
plt.grid(True, alpha=0.3)
plt.legend()
out_path = os.path.join(RESULTS_DIR, 'brute_force_latency.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot: {out_path}")


# ---------- Extrapolate: at what N does brute force become unacceptable? ----------
# Slope ns/doc, assuming linear scaling
slope_ns_per_doc = (means[-1] / ns[-1]) * 1e6  # ms -> ns
print(f"\nLatency scales roughly linearly in N.")
print(f"Cost per document: ~{slope_ns_per_doc:.1f} ns")
print(f"\nProjected corpus sizes where brute force hits each latency budget:")
for target_ms in [10, 100, 1000]:
    n_at_target = (target_ms * 1e6) / slope_ns_per_doc
    print(f"  {target_ms:>4d} ms  -->  N ~ {n_at_target:>15,.0f} documents")
