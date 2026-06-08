# Evidence Rank — Team BuriBuri

> **Redrob Hackathon** · Intelligent Candidate Discovery & Ranking Challenge  
# Evidence Rank — Team BuriBuri

> **Redrob Hackathon** · Intelligent Candidate Discovery & Ranking Challenge  
> Rank the top 100 Senior AI Engineer candidates from a 100,000-candidate pool — in under 5 seconds on CPU.

---

## Quick Start

### 1. Setup
* **Download the dataset:** You must provide the 100K candidate dataset. Download or copy your `candidates.jsonl` file and place it directly in the root directory of this repository (`./candidates.jsonl`).
* **Artifacts:** The heavy pre-computed artifacts (FAISS indexes, sparse matrices, cross-encoder scores) required to run the CPU ranker in under 5 minutes are already included in the `artifacts/` folder via Git LFS. You do not need to re-run the heavy GPU embeddings.

### 2. Run (Stage 3 Validator Command)
Run this single exact command to produce the final submission CSV from the candidates file within the 5-minute, 16GB, CPU-only constraint:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the CPU ranker
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv

# Validate format compliance
python validate_submission.py team_BuriBuri.csv
```

### 3. Running Tests
The unit test suite validates scoring formulas, re-ranking multipliers, location bands, reason generation, and sparse matrix arithmetic. Run all tests with:
```bash
pytest -v
```

> **Runtime:** ~2.2 sec · **RAM:** < 1 GB · **Compute:** CPU only · **Network:** None

---

## Architecture

The system utilizes an offline-online hybrid architecture designed to achieve sub-second execution on CPU at runtime while maintaining deep search capabilities. 

This design was finalized after multiple planning iterations, comprehensive retrieval-scoring experiments and deliberate engineering trade-offs between offline feature pre-computation and runtime compute constraints.

![Architecture Diagram](architecture_evidence-rank.png)

### Offline Pre-computation — `preprocess.py` (~57 min on T4 GPU)

**Phase 0 — JD Intelligence**  
The Job Description contract (`metadata/JD_contract.yaml`) is encoded using `BAAI/bge-m3` into dense + sparse query vectors. These become the retrieval anchors for the entire pipeline.

**Phase 1 — Corpus Embedding**  
All 100,000 candidate profiles are encoded with `BAAI/bge-m3`:
- Dense vectors → FAISS `IndexFlatIP` (cosine similarity, dim=1024)
- Sparse vectors → SciPy CSR matrix (learned lexical weights)
- BM25 index → tokenized profile text for lexical fallback

**Phase 1f — Honeypot Detection**  
All 100,000 candidates are scanned for impossible profiles before any ranking:
- Timeline contradictions (e.g., 8 years at a 3-year-old company)
- Skill impossibilities (expert in 10 skills with 0 years of use)
- Ghost profiles (no recent activity, no verifiable signals)
- Target-domain duration contradictions, such as expert Pinecone/search/retrieval claims that exceed claimed YoE plus a small buffer

Flagged candidates receive a score penalty that naturally pushes them below rank 100.

**Phase 1d — High-Recall RRF Retrieval**
Six signals are fused using Reciprocal Rank Fusion (RRF) to produce a wider shortlist of the top **15,000** candidates:
1. BGE-M3 dense cosine similarity
2. BGE-M3 sparse lexical similarity
3. BM25 keyword match
4. Title-query match
5. Skill-query match
6. Field-aware exact/regex recall over titles, skills, and career descriptions

The exact recall lane is CPU-cheap and intentionally conservative: career-history evidence is weighted more than skills-only claims, and candidates still need at least one retrieval/ranking/recommendation/evaluation signal.

**Phase 1c — Feature Extraction**  
The widened retrieval pool has 70+ features extracted by a handcrafted regex engine across three buckets:
- **Bucket A (Must-Haves):** Semantic search, vector DB, retrieval/ranking, NLP/LLM product experience
- **Bucket B (Nice-to-Haves):** MLOps, recommendations, evaluation pipelines, knowledge graphs
- **Bucket C (Career Quality):** Product company ratio, title progression, seniority, recency

**Phase 1e — Cross-Encoder Scoring**  
`BAAI/bge-reranker-v2-m3` scores the configured top retrieval candidates offline. The score is stored and merged at runtime, eliminating any GPU requirement at inference time. Candidates without a CE score fall back to their handcrafted core score.

---

### Online Ranking — `rank.py` (~2.3 sec on CPU)

**Phase 4 — Core Scoring**  
A vectorized Polars formula merges all precomputed features into a 0–100 `core_score`, blends in the cross-encoder score at a 65/35 handcrafted/CE ratio, then slices to the top 500 by blended Phase 4 score. This gives the semantic reranker enough influence to improve the NDCG-heavy top 10 while retaining JD-specific handcrafted evidence.

**Phase 5 — Behavioral Modifiers**  
Strict modifiers applied to the top 500, using 16 of 23 Redrob behavioral signals:

| Signal | How it's used |
|--------|--------------|
| `notice_period_days` | Notice modifier; extra risk for >90-day notice with weak eval evidence |
| `open_to_work_flag` | Availability penalty only when explicitly false |
| `recruiter_response_rate` | Responsiveness multiplier |
| `avg_response_time_hours` | Responsiveness multiplier |
| `interview_completion_rate` | Hiring intent signal |
| `offer_acceptance_rate` | Hiring intent signal |
| `github_activity_score` | Technical activity boost |
| `saved_by_recruiters_30d` | Market demand signal |
| `skill_assessment_scores` | Verified skill trust score |
| `endorsements_received` | Peer validation multiplier |
| `profile_completeness_score` | Base quality floor |
| `last_active_date` | Recency / ghost detection |
| `applications_submitted_30d` | Active job-seeking signal |
| `preferred_work_mode` | Location / remote fit |
| `willing_to_relocate` | Location / remote fit |
| `linkedin_connected` | Profile authenticity |

Phase 5 also applies Phase 1f/1c trust penalties. Target-skill duration contradictions are soft trust penalties because skill-duration metadata is noisy; hard impossible flags rely on contradictions visible in the candidate JSONL, such as copied role histories.

*Unused signals (7/23): `signup_date`, `profile_views_received_30d`, `connection_count`, `expected_salary_range_inr_lpa`, `search_appearance_30d`, `verified_email`, `verified_phone`*

**Phase 6 — Reasoning**  
For the final top 100, a 1–2 sentence reasoning string is assembled from actual extracted values in the candidate's profile. No LLM is called. Every claim corresponds to a real, verified field from the data.

---

## Evaluation Metrics

The submission is scored against a hidden ground truth using:

| Metric | Weight | What it measures |
|--------|--------|-----------------|
| NDCG@10 | **0.50** | Quality of top-10 picks |
| NDCG@50 | **0.30** | Quality of top-50 picks |
| MAP | **0.15** | Precision across all relevance levels |
| P@10 | **0.05** | Fraction of top-10 that are tier 3+ relevant |

**Final composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10**

> Honeypot rate > 10% in top 100 → automatic disqualification at Stage 3.

---

## Rerun Lifecycle

| Change made | What to rerun |
|-------------|--------------|
| `weights.yaml` only | `rank.py` only |
| `src/scorer.py`, `src/behavioral.py`, `src/explainer.py` | `rank.py` only |
| `src/features.py` or honeypot/contradiction logic | `preprocess.py --skip-embed` → `rank.py` |
| Exact/regex recall-lane changes in `preprocess.py` | `preprocess.py --skip-embed` → `rank.py` |
| `metadata/JD_contract.yaml` (signal/extraction terms) | `preprocess.py --skip-embed` → `rank.py` |
| Need CE scores for widened retrieval pool | `preprocess.py --only-cross-encoder' on GPU → `rank.py` |
| `metadata/JD_contract.yaml` (retrieval/query terms) | Full `preprocess.py` (GPU required) |
| Candidate data or embedding model | Full `preprocess.py` (GPU required) |

### Partial Rerun (Skip 57-minute Embedding)

```bash
python preprocess.py --candidates ./candidates.jsonl --skip-embed
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
```

Use this after recall-lane, feature, honeypot, or scoring-contract changes. It reuses existing BGE/FAISS/BM25 artifacts, widens `retrieval_scores.parquet` with the exact recall lane, then regenerates `candidate_features.parquet`.

The first partial rerun also creates `artifacts/retrieval_scores_base.parquet` so repeated recall-lane experiments start from the same dense/sparse/BM25 base instead of compounding old rescue outputs.

Run the full GPU preprocessing again only when JD embedding query text, the embedding model, candidate data, or dense/sparse/BM25 index construction changes.

### Cross-Encoder Refresh Only (GPU, No Re-Embedding)

After a widened retrieval/features run, refresh CE scores for the wider pool without rebuilding embeddings:

```bash
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
```

Use this on Colab if you want CE coverage for rescued candidates. It only rewrites `artifacts/cross_encoder_scores.parquet`.
evidence-rank/
├── preprocess.py              # Offline GPU pipeline (Phases 0, 1, 1f, 1d, 1c, 1e)
├── rank.py                    # Fast CPU ranker (Phases 4, 5, 6)
├── app.py                     # Gradio sandbox + CLI runner (no artifacts needed)
├── constants.py               # All file paths and model IDs
├── weights.yaml               # All scoring weights (tunable without rerun)
├── submission_metadata.yaml   # Hackathon submission metadata
├── requirements.txt           # All dependencies
│
├── src/
│   ├── features.py            # Phase 1c: regex feature extraction
│   ├── scorer.py              # Phase 4: core scoring formula
│   ├── behavioral.py          # Phase 5: behavioral modifiers
│   ├── explainer.py           # Phase 6: reason generation
│   ├── jd_intelligence.py     # Phase 0: JD query builder
│   └── reranker.py            # Cross-encoder scoring utility
│
├── metadata/
│   └── JD_contract.yaml       # Job Description signal contract (rubric)
│
└── artifacts/                 # Generated by preprocess.py
    ├── faiss_index.bin
    ├── candidate_sparse_matrix.npz
    ├── candidate_texts.pkl
    ├── bm25_index.pkl
    ├── retrieval_scores.parquet
    ├── candidate_features.parquet
    ├── cross_encoder_scores.parquet
    └── candidate_flags.parquet
```

---

## Sandbox Demo

Live heuristic sandbox on HuggingFace Spaces (no GPU, no artifacts):  
**https://huggingface.co/spaces/sathvik1610/evidence-rank**

### CLI (local, same logic):

```bash
python app.py --candidates ./sample_candidates.json --out output.csv
python app.py --candidates ./candidates.jsonl --out output.csv
python app.py --candidates ./candidates.jsonl.gz --out output.csv
```

### Gradio UI (local):

```bash
python app.py
# Open http://localhost:7860
```

The sandbox runs Phases 1f + 1c + 4 + 5 + 6 using pure heuristics. Dense retrieval and cross-encoder are disabled. No `artifacts/` folder needed.

---

## Submission Spec Compliance

- [x] Exactly 100 rows, header `candidate_id,rank,score,reasoning`
- [x] All candidate IDs exist in `candidates.jsonl`
- [x] Ranks 1–100 each appear exactly once
- [x] Scores monotonically non-increasing by rank
- [x] Ties broken by `candidate_id` ascending
- [x] UTF-8 CSV
- [x] `rank.py` runs in 2.3 seconds (well under 5-minute limit)
- [x] No GPU during ranking
- [x] No external API calls during ranking
- [x] 16/23 Redrob behavioral signals used
- [x] Honeypot detection implemented (Phase 1f)
- [x] Reasoning is specific, non-templated, hallucination-free
- [x] `submission_metadata.yaml` present at repo root
- [x] Working sandbox provided

---

## Final Run Commands

### Colab Setup

Use Python 3.10/3.11 with a T4 GPU runtime. **You must downgrade NumPy before installing requirements.** 
Google Colab pre-installs NumPy 2.x, which has major C-API changes. If you skip this step, `faiss-cpu` and `scipy` will crash with the error: *`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`*.

Run these exact commands in order:

```bash
pip uninstall -y numpy
pip install "numpy==1.26.4"
pip install -r requirements.txt
```

### Full GPU Preprocess

Run this only when candidate data, embedding model, JD retrieval query text, or dense/sparse/BM25 index construction changed.

```bash
python preprocess.py --candidates ./candidates.jsonl
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

### Lightweight Local Preprocess

Run this after feature, honeypot, contradiction, exact-recall, or scoring-contract changes. This does not rebuild BGE embeddings.

```bash
python preprocess.py --candidates ./candidates.jsonl --skip-embed
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

### Cross-Encoder Refresh Only

Run this on Colab after widening or changing the retrieval candidate pool. It only rewrites `artifacts/cross_encoder_scores.parquet`.

```bash
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```
