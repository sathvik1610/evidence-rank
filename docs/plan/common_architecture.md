## 1. Problem Understanding & Design Philosophy

### 1.1 What the hackathon is actually testing

The challenge is not asking you to find candidates who contain the most AI keywords. It is asking you to reason about candidates the way a skilled recruiter would. The JD for Senior AI Engineer at Redrob AI is unusually honest about what it wants and — more importantly — what it does not want.

The scoring weights confirm this: NDCG@10 carries 50% of the total score. Your top 10 picks matter more than everything else combined.

### 1.2 What the JD actually means

The JD explicitly tells participants: *"The right answer involves reasoning about the gap between what the JD says and what the JD means."*

A Tier 5 candidate may not use the words "RAG" or "Pinecone" in their profile, but if their career history shows they built a recommendation system at a product company, they are a fit. A candidate who has all the AI keywords listed as skills but whose title is "Marketing Manager" is not a fit.

**The ideal candidate (JD's own words):**
- 6-8 years total experience, 4-5 at product companies in applied ML roles
- Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale
- Has strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), LLM integration — and can defend them with reference to systems they actually built
- Located in or willing to relocate to Noida or Pune
- Active on the Redrob platform

**Hard disqualifiers (from JD text):**
- Pure research background with zero production deployment
- "AI experience" consisting primarily of LangChain/OpenAI wrappers with under 12 months of use and no pre-LLM ML background
- Senior engineer who has not written production code in the last 18 months
- Entire career at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, etc.) with zero product company exposure
- Computer vision, speech, or robotics specialists without significant NLP/IR exposure
- 5+ years entirely on closed-source proprietary systems without papers, talks, or open-source

**The JD's critical culture signal:**
The JD says: *"We work async-first and write a lot. If you find writing painful, you'll find this role painful."* Candidates with blank or minimal job descriptions are a cultural mismatch regardless of their skill set.

**The shipper vs researcher distinction:**
The JD explicitly says it tilts toward shippers over researchers. Candidates whose career language is full of "shipped," "launched," "deployed," "real users," "production" rank above equally skilled candidates whose language is "proposed," "benchmark," "ablation study," "novel architecture."

### 1.3 What the behavioral signals actually mean

The 23 `redrob_signals` fields are not profile quality metrics. They are **availability and reachability metrics**. A candidate with a perfect skills profile who last logged in 8 months ago and has a 5% recruiter response rate is, for practical hiring purposes, not a real option. Behavioral signals should be used as multiplicative modifiers on top of technical fit scores, not as primary ranking signals. The ground truth is scored on fit, not hireability — so behavioral signals should never override strong technical evidence.

**The ghost exception:** A candidate who is inactive for over 365 days AND has a response rate below 5% AND is not open to work AND has submitted zero applications in 30 days is a dead profile. They will never be hired regardless of fit. These can be safely pre-filtered.

---

## 2. System Architecture Overview

The system executes in two strictly separated environments:

**Offline Precompute (`preprocess.py`, no time limit, run once before submission — GPU/CPU adaptive):**
- Phase 0: Parse `JD_contract.yaml` → build 3 natural-language JD query strings → encode via BGE-M3 → save 3 dense `.npy` vectors + 1 learned-sparse `.npz` CSR matrix + BM25 keyword list. **ColBERT disabled entirely.**
- Phase 1: Encode all 100K candidate profile texts via BGE-M3 (dense + learned-sparse only) → save dense matrix as FAISS `IndexFlatIP` + save sparse outputs as a single consolidated `candidate_sparse_matrix.npz` CSR matrix
- Phase 1b: Build traditional `rank_bm25` lexical index on normalized candidate text fields
- **Phase 1f: Honeypot Audit Pass** — separate pure-JSON structural validation pass (no ML, no embeddings). Runs two-tier detection: `impossible_flag` (hard structural contradictions) and `honeypot_score` (0.0–1.0 soft weighted sum). Computes `suspicious_flag = honeypot_score > 0.70`. Saves all three to `candidate_flags.parquet`. **Must run after Phase 1 text encoding, before Phase 1d RRF scoring.**
- Phase 1d: Compute 6-way high-recall RRF retrieval scores offline; save top-15000 to `retrieval_scores.parquet`
- Phase 1c: Extract candidate features offline (Bucket A/B/C) on the widened retrieval pool; save to `candidate_features.parquet`
- Phase 1e: Run cross-encoder on the configured retrieval pool (`constants.CE_PRECOMPUTE_TOPK`, currently 15,000) offline; save to `cross_encoder_scores.parquet`

**Runtime Ranking Engine (`rank.py`, must complete in under 5 minutes on CPU, no network):**
- Input guard: read `candidate_id` values from `--candidates` and filter precomputed features to that exact input file. If there is no overlap, exit and require a fresh preprocess run.
- Phase 2: When `artifacts/retrieval_scores.parquet` exists, join retrieval scores and filter to the configured runtime pool from `weights.yaml` (`retrieval.runtime_top_k`, currently 10,000). Sample `--skip-embed` runs still score all sample candidates.
- Phase 3: Load precomputed candidate features from parquet (< 5s)
- Phase 4: Compute core score, merge precomputed cross-encoder scores, then slice top 500 by blended Phase 4 score (< 30s)
- Phase 5: Behavioral re-ranking + penalization → top 100 (< 10s)
- Phase 6: Reason generation → final CSV (< 30s)
- Phase 7: Manual validation (pre-submission, not part of 5-min constraint)

**Demo Sandbox (`app.py`, HuggingFace Spaces, no precomputed artifacts):**
- Heuristic-only BM25 path; no FAISS or cross-encoder at inference time
- Accepts a small candidate JSON payload; returns ranked results in under 10s
- Used for interactive demonstration only — not the competition submission path

### 2.1 Dataset Assumption

This system assumes the official competition dataset (`candidates.jsonl` as distributed in the hackathon bundle). Runtime embedding generation is **intentionally unsupported** — the competition explicitly permits offline precomputation, and adding fallback runtime ML inference would force heavy dependencies into `rank.py`.

**Small-file sample path:** Run `preprocess.py --sample --skip-embed` first. This creates flags and features for `sample_candidates.json` without dense embeddings, FAISS, BM25, RRF, or cross-encoder artifacts. Then `rank.py --candidates sample_candidates.json --out sample_submission.csv` ranks those precomputed sample features. If feature artifacts are missing, `rank.py` exits and asks for preprocess; it does not recompute Phase 3 at runtime.

**High-recall experiment path:** Run `preprocess.py --candidates ./candidates.jsonl --skip-embed` after changing exact recall, feature extraction, honeypot logic, or JD extraction terms. This reuses existing BGE/FAISS/BM25 artifacts, fuses the saved RRF list with the CPU-cheap exact recall lane, and regenerates features for the widened pool. Full GPU preprocessing is needed only when candidate data, embedding model, dense/sparse query text, or index construction changes.

**Weight Overfitting Protection:**
Validation must be performed on the 150-200 manually-labeled candidates without aggressively tuning weights to prevent validation set overfitting. Keep weights conservative and generalized.

---

## Execution Model

The system operates in two distinct stages:

### Stage A: Offline Precomputation

Executed once before ranking.

Purpose:

* Perform expensive computations that do not depend on a specific job description.

Examples:

* Candidate validation and filtering
* Feature extraction
* Behavioral signal extraction
* Dense embedding generation
* FAISS index construction
* BM25 index construction
* Candidate metadata generation

Outputs:

* candidate_flags.parquet
* FAISS indices
* BM25 indices
* Candidate feature artifacts
* Any other reusable ranking artifacts

Notes:

* May exceed 5 minutes.
* May use different hardware during development.
* Produces artifacts consumed by the ranking stage.

---

### Stage B: Ranking Execution

This is the step evaluated under competition constraints.

Inputs:

* Job Description
* Candidate dataset
* Precomputed artifacts

Process:

1. Build JD representations
2. Run retrieval (BM25 + Dense Retrieval)
3. Apply RRF
4. Score shortlisted candidates
5. Apply behavioral adjustments
6. Generate explanations
7. Output `team_BuriBuri.csv` or the `--out` path passed to `rank.py`

Constraints:

* ≤ 5 minutes wall-clock
* ≤ 16 GB RAM
* CPU only
* No external API calls

---

### Design Principle

Any computation that does not require the current Job Description should be moved to Stage A whenever possible.

Stage B should function as a lightweight ranking system operating primarily on precomputed features and indices.

---

## 3. Repository Layout

```
├── rank.py                        # Runtime entry point
├── preprocess.py                  # Offline pipeline runner
├── constants.py                   # Single source of truth for all artifact paths (see §3.1)
├── test_sparse_pipeline.py        # Step 1 math validation (must pass before Phase 1 implementation)
├── weights.yaml                   # Dynamic weights, multipliers, boosts, and thresholds config file
├── requirements.txt
├── submission_metadata.yaml
├── README.md
├── app.py                         # HuggingFace Spaces Gradio sandbox
├── src/
│   ├── __init__.py
│   ├── jd_intelligence.py          # Phase 0: YAML/JD-derived query, HyDE, and BM25 keyword generation
│   ├── features.py                # Phase 3: Bucket A/B/C extraction
│   ├── scorer.py                  # Phase 4: Weighted scoring formula
│   ├── reranker.py                # Phase 4: Cross-encoder reranking
│   ├── behavioral.py              # Phase 5: Behavioral multipliers, seniority, and penalties
│   └── explainer.py              # Phase 6: Reason generation
├── metadata/
│   └── validation_set.json        # Hand-labeled validation set (150-200 candidates)
└── artifacts/
    ├── jd_v1_skills.npy             # Phase 0: Dense query — v1_skills (programmatically built from YAML)
    ├── jd_hyde_recsys.npy           # Phase 0: Dense query — HyDE RecSys Persona
    ├── jd_hyde_eval.npy             # Phase 0: Dense query — HyDE Eval/Metrics Persona
    ├── jd_sparse_queries.npz        # Phase 0: Learned-sparse CSR query matrix (BGE-M3, 3 × vocab_size)
    ├── jd_keywords.json             # Phase 0: BM25 keyword list
    ├── jd_config.json               # Phase 0: Structured JD rule set
    ├── faiss_index.bin              # Phase 1: FAISS IndexFlatIP (dense candidate embeddings)
    ├── candidate_ids.json           # Phase 1: Ordered candidate ID list (matches FAISS row index)
    ├── candidate_sparse_matrix.npz  # Phase 1: Consolidated learned-sparse CSR matrix (100K × vocab_size)
    ├── candidate_texts.pkl          # Phase 1: Serialized normalized candidate texts for BM25
    ├── bm25_index.pkl               # Phase 1: Serialized rank_bm25 index
    ├── candidate_flags.parquet      # Phase 1f: impossible_flag (bool), honeypot_score (float), suspicious_flag (bool), is_ghost (bool), disqualifier tags
    ├── retrieval_scores_base.parquet # Phase 1d: Dense/sparse/BM25 base retrieval snapshot for repeatable recall experiments
    ├── retrieval_scores.parquet     # Phase 1d: Precomputed high-recall RRF scores (top 15000 candidates)
    ├── candidate_features.parquet   # Phase 1c: Precomputed Bucket A/B/C features for retrieved candidates
    └── cross_encoder_scores.parquet # Phase 1e: Precomputed bge-reranker-v2-m3 scores for the configured CE pool
```

**What each key file does:**

`rank.py` — Competition entry point. Takes `--candidates` and `--out` as CLI arguments. Reads input candidate IDs, filters precomputed features to that input file, optionally applies the RRF runtime cutoff if retrieval scores exist, then runs Phases 4–6 as pure in-memory operations and writes the output CSV. Must complete in under 5 minutes. **Does not import `flagembedding`, `sentence-transformers`, `faiss`, or `torch` at import time.**

`preprocess.py` — Offline pipeline runner. Phase 0 JD intelligence, Phase 1 corpus embedding + sparse CSR matrix build, FAISS index, BM25 index, honeypot flagging, ghost pre-filtering, 6-way high-recall RRF retrieval scoring, feature extraction, cross-encoder inference. No time constraint. GPU-adaptive for embedding steps.

`src/jd_intelligence.py` — Reads `metadata/JD_contract.yaml` and `job_description.txt`, then generates `jd_config.json`, BM25 keywords, dense query text, HyDE-style query text, and the cross-encoder JD query. This is the single source for Phase 0 retrieval text; do not reintroduce hardcoded JD/HyDE strings in `preprocess.py`.

`constants.py` — Single source of truth for every artifact path string. Both `preprocess.py` and `rank.py` (and all modules under `src/`) import paths exclusively from here. Renaming any artifact requires one edit in one file. See §3.1.

`test_sparse_pipeline.py` — Step 1 validation script. Runs a 9-step end-to-end proof of the sparse storage and dot-product math on 1,000 mock candidates before any other pipeline code is written. **Must pass before starting Phase 1 implementation.**

`weights.yaml` — Dynamic configuration file containing all scoring weights, multipliers, boosts, and thresholds. Loaded at startup by both pipelines to avoid hardcoded parameter logic.

`app.py` — HuggingFace Spaces demo sandbox. Heuristic BM25-only path; no transformer model loads. Accepts small JSON payloads for interactive demonstration.

Retrieval lives in `preprocess.py`. At preprocess time: 6-way RRF (3 dense FAISS searches + 1 learned-sparse CSR dot-product + 1 BM25 + 1 exact/regex recall lane) → saves `retrieval_scores.parquet`. At rank time: loads parquet, filters to top N.

`src/features.py` — At preprocess time: YAML-contract-driven Bucket A/B/C extraction on all retrieved candidates → saves `candidate_features.parquet`. At rank time: loads parquet.

`src/scorer.py` — Weighted formula from `weights.yaml`: must-have (55%), nice-to-have (5%), career quality (15%), product builder (25%). Pure pandas/numpy; no model calls.

`src/reranker.py` — At preprocess time only: cross-encoder scoring using `bge-reranker-v2-m3` on the configured CE pool → saves `cross_encoder_scores.parquet`. At rank time: loads parquet, merges scores using `weights.yaml`.

`src/behavioral.py` — Multiplicative modifiers: availability, notice, YAML-derived location bands, social proof, seniority, writing signal, soft penalties, floor-exempt disqualifier handling, and 90-day bonus.

`src/explainer.py` — Deterministic reason generation from profile facts and extracted evidence snippets. It does not call an LLM, guess facts, or invent skills/durations.

---

### 3.1 Design Rule: Single Source of Truth for Artifact Paths

**Problem this solves:** The offline precompute script saves an artifact to one filename; the runtime loader looks for a slightly different filename. The mismatch is silent — `FileNotFoundError` only surfaces at evaluation time. With 25+ days of development across multiple files, this class of bug will happen without a structural guard.

**Rule:** Every artifact path string is defined **once** in `constants.py`. Both `preprocess.py` and every module under `src/` import their paths from `constants.py`. No path string is ever written inline in a script.

The canonical constant definitions (do not duplicate these elsewhere in the codebase):

```python
# constants.py — single source of truth for all artifact paths
# Both preprocess.py (write side) and rank.py / src/* (read side) import from here.
# To rename an artifact: change it in exactly one place.

# --- Phase 0: JD query artifacts ---
JD_DENSE_PATHS = {
    "v1_skills":   "artifacts/jd_v1_skills.npy",
    "hyde_recsys": "artifacts/jd_hyde_recsys.npy",
    "hyde_eval":   "artifacts/jd_hyde_eval.npy",
}
JD_SPARSE_PATH  = "artifacts/jd_sparse_queries.npz"  # 3 × vocab_size CSR matrix
JD_KEYWORDS_PATH = "artifacts/jd_keywords.json"
JD_CONFIG_PATH   = "artifacts/jd_config.json"

# --- Phase 1: Candidate corpus artifacts ---
FAISS_INDEX_PATH       = "artifacts/faiss_index.bin"
CANDIDATE_IDS_PATH     = "artifacts/candidate_ids.json"
CANDIDATE_SPARSE_PATH  = "artifacts/candidate_sparse_matrix.npz"  # 100K × vocab_size CSR
BM25_INDEX_PATH        = "artifacts/bm25_index.pkl"
BM25_TEXTS_PATH        = "artifacts/candidate_texts.pkl"
CANDIDATE_FLAGS_PATH   = "artifacts/candidate_flags.parquet"

# --- Phase 1c / 1d / 1e: Precomputed scoring artifacts ---
RETRIEVAL_SCORES_PATH      = "artifacts/retrieval_scores.parquet"
CANDIDATE_FEATURES_PATH    = "artifacts/candidate_features.parquet"
CROSS_ENCODER_SCORES_PATH  = "artifacts/cross_encoder_scores.parquet"
```

**Usage pattern** — every write and every read goes through the constant:

```python
# preprocess.py (write side)
from constants import CANDIDATE_SPARSE_PATH, JD_SPARSE_PATH
scipy.sparse.save_npz(CANDIDATE_SPARSE_PATH, candidate_csr)
scipy.sparse.save_npz(JD_SPARSE_PATH, query_csr)

# preprocess.py (read side for retrieval fusion)
from constants import CANDIDATE_SPARSE_PATH, JD_SPARSE_PATH, FAISS_INDEX_PATH
candidate_csr = scipy.sparse.load_npz(CANDIDATE_SPARSE_PATH)
jd_sparse     = scipy.sparse.load_npz(JD_SPARSE_PATH)
index         = faiss.read_index(FAISS_INDEX_PATH)
```

**Enforcement:** When writing any new file I/O call, check `constants.py` first. If the path isn't there yet, add it to `constants.py` before referencing it anywhere else. Never use a string literal for an artifact path inside `preprocess.py`, `rank.py`, or any `src/` module.

---

