"""
hello_embeddings.py — first taste of embeddings.

We give the computer 4 sentences. It has never seen them before.
It turns each sentence into a list of 384 numbers.
We then ask: how similar is each pair?

If the 2 gardening sentences score high, and the 2 finance sentences score high,
and the cross-pairs (gardening <-> finance) score low,
then the computer has figured out the *meaning* without being told anything
about what gardening or finance is. That's the magic.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


# Step 1 — load the model. Small, 384-dimensional, ~80 MB. Runs on CPU.
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2 — our 4 sentences. 2 about gardening, 2 about finance.
sentences = [
    "I love plants",                  # gardening
    "Gardening is fun",               # gardening
    "The stock market crashed",       # finance
    "Bonds and treasury yields",      # finance
]

# Step 3 — turn each sentence into a vector (list of 384 numbers).
print("Embedding sentences...")
vectors = model.encode(sentences)
print(f"Shape: {vectors.shape}  (4 sentences, 384 numbers each)\n")


# Step 4 — define cosine similarity.
# Range: -1 (opposite) ... 0 (unrelated) ... 1 (identical direction).
# In practice for text embeddings: ~0 means "unrelated", >0.4 means "related".
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Step 5 — score every pair and print.
print("Similarity between each pair (higher = more similar in meaning):\n")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine(vectors[i], vectors[j])
        marker = "  <-- HIGH" if sim > 0.4 else ""
        print(f"  {sim:.3f}   '{sentences[i]}'  vs  '{sentences[j]}'{marker}")
