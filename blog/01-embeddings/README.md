# Post 1 — Embeddings & Vector Search Internals

**Status:** in progress. Target: first published post.

## The question this post answers

Given a corpus of millions of documents and a query, return the top-k nearest neighbors in <10ms with recall@10 > 0.95, on a laptop. How does that actually work?

## Sections (draft)

1. What an embedding is — intrinsic vs extrinsic geometry, the basis-free view, why "similarity" is a *choice* not a given
2. Similarity metrics — cosine, dot product, L2. When each is correct. The unit-normalization trick and what it costs.
3. Brute-force baseline — naive O(n) search, vectorized numpy, measured runtime as a function of n and d
4. The curse of dimensionality — empirical: as d grows, what happens to the distribution of distances? Plot.
5. IVF (Inverted File Index) — k-means clustering, probe count, recall/latency tradeoff. From scratch.
6. PQ (Product Quantization) — block decomposition, codebook learning, asymmetric distance computation. From scratch.
7. HNSW (Hierarchical Navigable Small World) — multi-layer graph, greedy descent, M and efSearch. From scratch.
8. Comparison — IVF vs PQ vs HNSW vs FAISS reference, measured: index build time, query latency, recall@10, memory footprint
9. Production considerations — when to use what; hosted (Pinecone / Weaviate / Qdrant) vs in-process

## Reading list

- Malkov & Yashunin, 2016 — *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (HNSW paper)
- Jégou, Douze, Schmid, 2010 — *Product quantization for nearest neighbor search*
- Johnson, Douze, Jégou, 2017 — *Billion-scale similarity search with GPUs* (FAISS)
- FAISS wiki — Getting Started + Index Types
- Pinecone / Qdrant engineering blogs on HNSW in production

## Dataset

- MS MARCO passage corpus (~8.8M short passages) — too big for brute-force on laptop; right size to show ANN wins
- For dev iteration: SciDocs (~300K) or a 100K subsample of MS MARCO

## Experiments to run (the numbers the post lives or dies on)

- recall@10 vs query latency for brute-force / IVF / PQ / HNSW / FAISS — single plot
- Index build time vs N for each
- Memory footprint vs N for each
- HNSW ablation: recall vs efSearch at fixed M; recall vs M at fixed efSearch — two plots
- Curse of dimensionality plot: distance distribution at d = 10, 100, 1000

## Assignment notebook deliverable

A reader works through `assignment.ipynb` and implements, from scaffolded TODOs:
- Brute-force NN with cosine
- IVF with k-means clustering
- HNSW with greedy descent

Tests check correctness against a small known-answer dataset and against the FAISS reference implementation.

## What would make this post fail (i.e. read as AI-demo, not pro depth)

- Hand-waving why HNSW works (the small-world property must be derived, not asserted)
- Skipping the recall/latency curves and just claiming "HNSW is faster"
- Using FAISS as the implementation rather than implementing the algorithm from scratch
- Showing only the success case; not documenting the bugs hit, the configs that didn't work, and the surprises
