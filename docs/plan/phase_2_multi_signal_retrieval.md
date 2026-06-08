## 6. Phase 2 — Multi-Signal Retrieval

### 6.1 What this phase does

**At preprocess time (`preprocess.py`):** Runs FAISS dense search, learned sparse search, BM25 sparse search, and a CPU-cheap exact/regex recall lane against all 100K candidates. Saves ranked candidate IDs and RRF scores to `artifacts/retrieval_scores.parquet`.

**At rank time (`rank.py`):** Loads `retrieval_scores.parquet` (using Polars for memory safety and speed), filters to top N by RRF score, then applies core scoring. **No FAISS, BM25, regex corpus scan, or model inference happens at rank time.**

**Retrieval pool size tuning:** Precompute saves the top 15,000 candidates to `retrieval_scores.parquet`. At runtime, this branch filters to top N = 10,000 via `weights.yaml` (`retrieval.runtime_top_k`). The wider pool is intentional: validation showed strong candidates can be lost before feature extraction, and those losses are unrecoverable later.

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

### 6.5 Six-Way Reciprocal Rank Fusion

Upgraded from 5-signal to **6-signal RRF** with `k=60` (industry standard default, tunable in `weights.yaml`).

The six orthogonal signals:
1. **Dense List 1** — Cosine similarity via FAISS against `jd_v1_skills.npy` (YAML-derived skills query)
2. **Dense List 2** — Cosine similarity via FAISS against `jd_hyde_recsys.npy` (RecSys HyDE persona)
3. **Dense List 3** — Cosine similarity via FAISS against `jd_hyde_eval.npy` (Eval/Metrics HyDE persona)
4. **Learned Sparse List 4** — C-speed dot-product via SciPy CSR across all YAML-derived sparse query rows, collapsed to one sparse ranking list by per-candidate max score
5. **Lexical Sparse List 5** — `rank_bm25` exact term matching scores
6. **Exact Recall List 6** — field-aware regex scan over current title, career titles/descriptions, summaries, and skills

**Tokenizer alignment rule:** The same lowercasing + punctuation-stripping function must be used for both the candidate text preprocessing loop (Phase 1b BM25 build) and the BM25 query construction. Mismatched tokenization silently degrades sparse recall.

**Exact recall guardrails:** This lane requires at least one primary retrieval/ranking/recommendation/evaluation signal. Career-description evidence is weighted highest; skills-only claims are allowed only when corroborated by multiple signals. Ghost IDs are excluded before fusion.

```python
import numpy as np
import scipy.sparse
import faiss
import json
from rank_bm25 import BM25Okapi

# --- Load precomputed artifacts ---
index = faiss.read_index("artifacts/faiss_index.bin")
all_ids = json.load(open("artifacts/candidate_ids.json"))
candidate_sparse_csr = scipy.sparse.load_npz("artifacts/candidate_sparse_matrix.npz")

# Load 3 dense JD query vectors
jd_v1 = np.load("artifacts/jd_v1_skills.npy").astype(np.float32).reshape(1, -1)
jd_recsys = np.load("artifacts/jd_hyde_recsys.npy").astype(np.float32).reshape(1, -1)
jd_eval = np.load("artifacts/jd_hyde_eval.npy").astype(np.float32).reshape(1, -1)

# Load sparse query CSR (row 0 = v1_skills, row 1 = recsys, row 2 = eval)
jd_sparse_all = scipy.sparse.load_npz("artifacts/jd_sparse_query.npz")

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
if jd_sparse_all.shape[1] < vocab_size:
    jd_sparse_all = scipy.sparse.hstack([
        jd_sparse_all,
        scipy.sparse.csr_matrix((jd_sparse_all.shape[0], vocab_size - jd_sparse_all.shape[1]))
    ])
elif jd_sparse_all.shape[1] > vocab_size:
    jd_sparse_all = jd_sparse_all[:, :vocab_size]

sparse_scores = candidate_sparse_csr.dot(jd_sparse_all.T).toarray().max(axis=1)
# .toarray() is mandatory because dot() returns a sparse structure.
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

# --- 6-Way RRF ---
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
    exact_recall_ids,    # Signal 6
])

# Sort by RRF score, save top 15000 to parquet
retrieved = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:15000]
retrieved_ids = [r[0] for r in retrieved]
```

**k-tuning note:** `k=60` is the industry default. If validation shows precision loss at top-10, try sweeping k ∈ [10, 100] against `metadata/validation_set.json`. Document the winning value in `weights.yaml`.

The widened retrieval pool goes to Phase 3 for feature extraction. Missing cross-encoder scores are allowed at runtime; those candidates fall back to handcrafted `core_score`.

**Recall verification (do this once during development):** After building the retrieval pool, compare against `metadata/validation_set.json`. A missing Tier 3 candidate at this stage is unrecoverable. Fix retriever thresholds before tuning post-processing.

### 6.6 Partial rerun guidance

Use:

```bash
python preprocess.py --candidates ./candidates.jsonl --skip-embed
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
```

This reuses existing dense/sparse/BM25 artifacts and recomputes only flags, high-recall rescue fusion, and features. Run full GPU preprocessing only after changing candidate data, embedding model, Phase 0 dense/sparse query text, or index construction.

To refresh semantic reranker coverage for the widened pool without re-embedding, run on GPU:

```bash
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
```

---

