## 6. Phase 2 — Multi-Signal Retrieval

### 6.1 What this phase does

**At preprocess time (`preprocess.py`):** Runs FAISS dense search, BM25 sparse search, and RRF fusion against all 100K candidates. Saves ranked candidate IDs and RRF scores to `artifacts/retrieval_scores.parquet`. This is a one-time operation.

**At rank time (`rank.py`):** Loads `retrieval_scores.parquet` (using Polars for memory safety and speed), filters to top N by RRF score, optionally applies soft activity boost from `candidate_flags.parquet`. **No FAISS or BM25 calls at rank time.**

**Retrieval pool size tuning:** Precompute saves the top 5,000 candidates to `retrieval_scores.parquet`. At runtime, we filter this down to the top N = 3,000 (controlled by `pool_size` in `weights.yaml`). Top 5,000 are saved to allow flexibility — Phase 2 slices to 3,000 at runtime, preserving the option to adjust the cutoff without re-running offline phases. Experiment with 2,000–3,000; the RRF-ranked pool drops off steeply in quality after the top 2,000, and a smaller pool reduces Phase 4 scoring time. Only increase beyond 3,000 if validation shows recall loss (Tier 3+ candidates missing from the Phase 4 input — check against `metadata/validation_set.json`).

Target rank-time cost: under 5 seconds on CPU.

### 6.2 Dense retrieval via FAISS

```python
import faiss
import numpy as np
import json

index = faiss.read_index("artifacts/faiss_index.bin")
all_ids = json.load(open("artifacts/candidate_ids.json"))

jd_skills_vec = np.load("artifacts/jd_skills_vector.npy").astype(np.float32).reshape(1, -1)
jd_ideal_vec = np.load("artifacts/jd_ideal_vector.npy").astype(np.float32).reshape(1, -1)

# Search both JD vectors
scores_skills, idx_skills = index.search(jd_skills_vec, k=2000)
scores_ideal, idx_ideal = index.search(jd_ideal_vec, k=2000)

dense_ids_skills = [all_ids[i] for i in idx_skills[0]]
dense_ids_ideal = [all_ids[i] for i in idx_ideal[0]]
```

### 6.3 Sparse retrieval via BM25

```python
import pickle
from rank_bm25 import BM25Okapi

with open("artifacts/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

keywords = json.load(open("artifacts/jd_keywords.json"))
query_tokens = " ".join(keywords).lower().split()

scores_bm25 = bm25.get_scores(query_tokens)
top_bm25_idx = np.argsort(scores_bm25)[::-1][:2000]
sparse_ids = [all_ids[i] for i in top_bm25_idx]
```

### 6.4 Soft activity boost

Before RRF, lightly boost candidates who are actively seeking. This prevents active candidates from being buried behind stale ones when ranks are equal.

```python
# Load lightweight signals from flags parquet
# For candidates in the top pools: if open_to_work OR last_active < 90d,
# artificially boost their retrieval rank by 200 positions (within the retrieval step only).
# This is a soft nudge, not a hard filter.
```

### 6.5 Five-Way Reciprocal Rank Fusion

Upgraded from 3-signal to **5-signal RRF** with `k=60` (industry standard default, tunable in `weights.yaml`).

The five orthogonal signals:
1. **Dense List 1** — Cosine similarity via FAISS against `jd_v1_skills_vector.npy` (YAML-derived skills query)
2. **Dense List 2** — Cosine similarity via FAISS against `jd_hyde_recsys_vector.npy` (RecSys HyDE persona)
3. **Dense List 3** — Cosine similarity via FAISS against `jd_hyde_eval_vector.npy` (Eval/Metrics HyDE persona)
4. **Learned Sparse List 4** — C-speed dot-product via SciPy CSR: `candidate_sparse_csr.dot(query_sparse_row.T).toarray().flatten()`
5. **Lexical Sparse List 5** — `rank_bm25` exact term matching scores

**Tokenizer alignment rule:** The same lowercasing + punctuation-stripping function must be used for both the candidate text preprocessing loop (Phase 1b BM25 build) and the BM25 query construction. Mismatched tokenization silently degrades sparse recall.

```python
import numpy as np
import scipy.sparse
import faiss
import json
from rank_bm25 import BM25Okapi

# --- Load precomputed artifacts ---
index = faiss.read_index("artifacts/faiss_index.bin")
all_ids = json.load(open("artifacts/candidate_ids.json"))
candidate_sparse_csr = scipy.sparse.load_npz("artifacts/candidate_sparse.npz")

# Load 3 dense JD query vectors
jd_v1 = np.load("artifacts/jd_v1_skills_vector.npy").astype(np.float32).reshape(1, -1)
jd_recsys = np.load("artifacts/jd_hyde_recsys_vector.npy").astype(np.float32).reshape(1, -1)
jd_eval = np.load("artifacts/jd_hyde_eval_vector.npy").astype(np.float32).reshape(1, -1)

# Load sparse query CSR (row 0 = v1_skills, row 1 = recsys, row 2 = eval)
jd_sparse_all = scipy.sparse.load_npz("artifacts/jd_sparse_query.npz")
# Use the v1_skills sparse row as the canonical sparse query signal
jd_sparse_row = jd_sparse_all[0]  # shape: (1, vocab_size)

# --- Signal 1, 2, 3: Dense FAISS searches ---
_, idx1 = index.search(jd_v1, k=2000)
_, idx2 = index.search(jd_recsys, k=2000)
_, idx3 = index.search(jd_eval, k=2000)
dense_ids_v1 = [all_ids[i] for i in idx1[0]]
dense_ids_recsys = [all_ids[i] for i in idx2[0]]
dense_ids_eval = [all_ids[i] for i in idx3[0]]

# --- Signal 4: Learned Sparse dot-product (C-speed) ---
# vocab_size must match the CSR matrix shape. Resize query if needed.
vocab_size = candidate_sparse_csr.shape[1]
if jd_sparse_row.shape[1] < vocab_size:
    jd_sparse_row = scipy.sparse.hstack([
        jd_sparse_row,
        scipy.sparse.csr_matrix((1, vocab_size - jd_sparse_row.shape[1]))
    ])
elif jd_sparse_row.shape[1] > vocab_size:
    jd_sparse_row = jd_sparse_row[:, :vocab_size]

sparse_scores = candidate_sparse_csr.dot(jd_sparse_row.T).toarray().flatten()
# .toarray().flatten() is mandatory — dot() returns a sparse structure
top_sparse_idx = np.argsort(sparse_scores)[::-1][:2000]
sparse_ids_learned = [all_ids[i] for i in top_sparse_idx]

# --- Signal 5: BM25 lexical ---
with open("artifacts/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)
keywords = json.load(open("artifacts/jd_keywords.json"))
query_tokens = normalize_text(" ".join(keywords)).split()  # same tokenizer as index build
bm25_scores = bm25.get_scores(query_tokens)
top_bm25_idx = np.argsort(bm25_scores)[::-1][:2000]
sparse_ids_bm25 = [all_ids[i] for i in top_bm25_idx]

# --- 5-Way RRF ---
def rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores = {}
    for ranked in ranked_lists:
        for rank_idx, cand_id in enumerate(ranked):
            scores[cand_id] = scores.get(cand_id, 0.0) + 1.0 / (k + rank_idx + 1)
    return scores

rrf_scores = rrf([
    dense_ids_v1,       # Signal 1
    dense_ids_recsys,   # Signal 2
    dense_ids_eval,     # Signal 3
    sparse_ids_learned, # Signal 4
    sparse_ids_bm25,    # Signal 5
])

# Sort by RRF score, save top 5000 to parquet
retrieved = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5000]
retrieved_ids = [r[0] for r in retrieved]
```

**k-tuning note:** `k=60` is the industry default. If validation shows precision loss at top-10, try sweeping k ∈ [10, 100] against `metadata/validation_set.json`. Document the winning value in `weights.yaml`.

The union of five retrievers typically produces 4,000–7,000 unique candidates before the top-5000 cut. These go to Phase 3 for feature extraction.

**Recall verification (do this once during development):** After building the retrieval pool, manually confirm that 10+ Tier 3 candidates from your validation set are present. A missing Tier 3 candidate at this stage is unrecoverable. Fix retriever thresholds before proceeding.

---

