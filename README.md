# Evidence Rank - Team BuriBuri

Redrob Hackathon submission for the Intelligent Candidate Discovery and Ranking Challenge.

Team BuriBuri: Sathvik Pilyanam, Pranathi Mandadi

The system ranks the best 100 candidates for the Redrob Senior AI Engineer role. It uses expensive offline preprocessing once, then runs the final competition ranking step quickly on CPU using precomputed artifacts.

## Main Run Commands

### Clone And Set Up

```bash
git clone https://github.com/sathvik1610/evidence-rank.git
cd evidence-rank

pip install -r requirements.txt
```

If the repository is stored with Git LFS artifacts, pull them after cloning:

```bash
git lfs install
git lfs pull
```

Place the official dataset at the repository root:

```text
candidates.jsonl
```

### Generate Final Submission

Use this command for the normal competition run. It does not rebuild embeddings; it uses the already precomputed artifacts in `artifacts/`.

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

Expected behavior:

- Output file: `team_BuriBuri.csv`
- Required columns: `candidate_id,rank,score,reasoning`
- Runtime: about 2-3 seconds locally with the current artifacts
- Ranking step: CPU only, no network calls, no embedding model loaded

### Run Tests

```bash
python -m pytest tests -q
```

Useful focused checks:

```bash
python -m pytest tests/test_features.py tests/test_scorer.py tests/test_behavioral.py tests/test_explainer.py -q
python -m py_compile preprocess.py rank.py src/features.py src/scorer.py src/behavioral.py src/explainer.py
```

## Important Artifact Note

Most expensive work has already been calculated and stored in `artifacts/`.

The final `rank.py` command depends on these precomputed files:

- `artifacts/candidate_features.parquet`
- `artifacts/retrieval_scores.parquet`
- `artifacts/cross_encoder_scores.parquet`
- `artifacts/candidate_flags.parquet`
- `artifacts/run_metadata.json`

Do not expect `rank.py` to rebuild missing features or embeddings. If feature artifacts are stale or missing, run `preprocess.py` first.

## When To Recalculate Preprocessing

Use this table before changing code. It explains what must be recalculated.

| Change made | Command to run |
|---|---|
| Only `weights.yaml` changed | `python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv` |
| `src/scorer.py`, `src/behavioral.py`, or `src/explainer.py` changed | `python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv` |
| `src/features.py` changed | `python preprocess.py --candidates ./candidates.jsonl --skip-embed`, then `rank.py` |
| Honeypot, contradiction, ghost, or exact recall logic in `preprocess.py` changed | `python preprocess.py --candidates ./candidates.jsonl --skip-embed`, then `rank.py` |
| `metadata/JD_contract.yaml` changed for feature patterns, penalties, city policy, or exact recall terms | `python preprocess.py --candidates ./candidates.jsonl --skip-embed`, then `rank.py` |
| Retrieval query text, embedding model, candidate data, FAISS/BM25/sparse index construction changed | Full `python preprocess.py --candidates ./candidates.jsonl`, then `rank.py` |
| Cross-encoder scores need refresh for the current retrieval pool | `python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder`, then `rank.py` |

### Lightweight Preprocess

Use this after feature, honeypot, contradiction, exact recall, or JD-contract extraction changes. It reuses existing BGE-M3 embeddings, FAISS index, sparse matrix, and BM25 index.

```bash
python preprocess.py --candidates ./candidates.jsonl --skip-embed
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

### Full GPU Preprocess

Use this only when candidate data, embedding model, dense/sparse JD query text, or index construction changed.

```bash
python preprocess.py --candidates ./candidates.jsonl
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

Full preprocessing loads BGE-M3 and the cross-encoder. It is meant for GPU/Colab-style runs, not for the final CPU-only ranking step.

### Cross-Encoder Refresh Only

Use this when the retrieval pool already exists but semantic reranker scores should be regenerated.

```bash
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

## Architecture Overview

The project is split into two stages.

### Stage A: Offline Preprocessing

Entry point: `preprocess.py`

This stage can use GPU and can take longer than the final ranking limit. It reads the full candidate dataset and creates reusable artifacts.

Phases:

1. Phase 0 - JD Intelligence
   - Reads `metadata/JD_contract.yaml` and `job_description.txt`.
   - Builds structured JD config, BM25 keywords, three dense JD query vectors, and learned-sparse JD query vectors.
   - Uses `BAAI/bge-m3`.
   - Disables ColBERT vectors to avoid huge memory usage.

2. Phase 1 - Corpus Preprocessing
   - Serializes each candidate profile into searchable text.
   - Builds dense BGE-M3 embeddings.
   - Stores dense vectors in FAISS `IndexFlatIP`.
   - Stores learned-sparse vectors in a SciPy CSR matrix.
   - Builds a BM25 index over normalized candidate text.

3. Phase 1f - Honeypot and Trust Checks
   - Reads raw JSON fields directly.
   - Flags impossible timelines, suspicious profiles, ghost profiles, consulting-only careers, research-only profiles, wrong-domain profiles, and skill-duration contradictions.
   - Saves results to `artifacts/candidate_flags.parquet`.

4. Phase 1d - Multi-Signal Retrieval
   - Builds a high-recall candidate pool using Reciprocal Rank Fusion.
   - Uses dense FAISS, learned-sparse BGE-M3, BM25, and exact/regex recall.
   - Saves retrieval scores to `artifacts/retrieval_scores.parquet`.

5. Phase 1c - Feature Extraction
   - Extracts JD-specific feature columns using `src/features.py`.
   - Uses YAML-driven patterns from `metadata/JD_contract.yaml`.
   - Saves `artifacts/candidate_features.parquet`.

6. Phase 1e - Cross-Encoder Scoring
   - Scores the configured retrieval pool with `BAAI/bge-reranker-v2-m3`.
   - Saves `artifacts/cross_encoder_scores.parquet`.

### Stage B: Runtime Ranking

Entry point: `rank.py`

This is the final evaluated path. It must stay fast, CPU-only, and artifact-driven.

Runtime flow:

1. Reads candidate IDs from `--candidates`.
2. Loads `artifacts/candidate_features.parquet`.
3. Filters features to candidates present in the input file.
4. Joins `artifacts/retrieval_scores.parquet` and keeps the runtime retrieval pool configured by `weights.yaml`.
5. Computes Phase 4 core score with `src/scorer.py`.
6. Merges precomputed cross-encoder scores with `src/reranker.py`.
7. Keeps top 500 by Phase 4 blended score.
8. Applies behavioral modifiers and penalties with `src/behavioral.py`.
9. Assigns deterministic ranks by descending score, then candidate ID.
10. Generates factual 1-2 sentence explanations with `src/explainer.py`.
11. Writes the final CSV and debug trace.

`rank.py` does not import `torch`, `faiss`, `FlagEmbedding`, or `sentence-transformers`. That is intentional.

## Methods Used

### BGE-M3 Dense Retrieval

`BAAI/bge-m3` encodes candidate profile text and JD query text. Candidate dense vectors are L2-normalized and stored in FAISS. FAISS inner product search is then equivalent to cosine similarity.

### BGE-M3 Learned-Sparse Retrieval

BGE-M3 also returns lexical weights. These are stored as sparse CSR matrices. Sparse candidate vectors are compared with sparse JD query vectors using fast matrix dot products.

### BM25 Retrieval

BM25 is used as a lexical anchor. It helps recover candidates who use exact domain terms such as BM25, FAISS, Pinecone, ranking, retrieval, NDCG, or evaluation.

### Exact/Regex Recall Lane

The exact recall lane scans titles, summaries, career descriptions, and skills with JD-contract patterns. It is intentionally conservative:

- career-description evidence is weighted more than skills-only evidence
- at least one primary retrieval/ranking/recommendation/evaluation signal is required
- ghost candidates are excluded from this lane

### Reciprocal Rank Fusion

Retrieval lists are fused with RRF using `retrieval.rrf_k` from `weights.yaml`. Preprocessing saves the widened top retrieval pool. Runtime slices it using `retrieval.runtime_top_k`.

### Feature Extraction

`src/features.py` converts candidate JSON into scoring features:

- Bucket A: retrieval/search, vector DB, evaluation, LTR/reranking, Python, LLM/RAG, distributed systems, HR-tech exposure
- Bucket B: product-company ratio, deployment language, shipper language, ownership, recency, depth, career IR density, isolated-template risk
- Bucket C: title velocity, consulting risk, keyword stuffing, stopped-coding risk, LangChain-only risk, closed-source risk

Feature extraction also copies profile facts and snippets into the feature parquet so the explainer can generate factual reasoning later.

### Honeypot Detection

Honeypot checks are rule-based and JSON-field based. They do not rely on embeddings. They look for:

- end date before start date
- negative role durations
- impossible years of experience
- skill durations wildly exceeding career timeline
- copied long role descriptions across multiple employers
- multiple current roles
- suspiciously maxed behavioral signals
- senior profiles with zero technical activity
- target-domain skill-duration overclaims

Hard impossible or suspicious profiles receive a severe score multiplier. Softer contradictions become trust penalties.

### Core Scoring

`src/scorer.py` computes a 0-100 technical score using weights from `weights.yaml`:

- must-have evidence
- nice-to-have evidence
- career quality
- product-builder score

It also adds manual-audit corrections for retrieval + LTR + evaluation strength, sustained career IR density, and isolated-template risk.

### Cross-Encoder Reranking

`src/reranker.py` merges offline `BAAI/bge-reranker-v2-m3` scores with the handcrafted core score. Current blend comes from `weights.yaml`:

- handcrafted score: 65%
- cross-encoder score: 35%

If a candidate has no cross-encoder score, runtime falls back to the handcrafted core score.

### Behavioral Reranking

`src/behavioral.py` applies late-stage modifiers for reachability and fit:

- last active date
- open-to-work flag
- recruiter response rate
- notice period
- location and relocation
- seniority
- writing signal
- social proof
- GitHub activity
- saved by recruiters
- interview completion
- offer acceptance
- profile completeness
- LinkedIn connection

Behavioral signals are late modifiers, not primary retrieval signals. Strong technical candidates are not removed early only because they are passive.

### Reason Generation

`src/explainer.py` generates the `reasoning` column for the final CSV.

Rules:

- no LLM is used
- no guessing
- title, company, and years of experience come from profile fields
- evidence snippets come from extracted candidate text snippets
- skill-duration contradiction flags prevent blind duration claims
- output is short, human-readable, and factual

## Repository Layout

```text
evidence-rank/
|-- README.md
|-- rank.py                         # Stage B final ranking entry point
|-- preprocess.py                   # Stage A offline artifact builder
|-- app.py                          # Demo sandbox
|-- constants.py                    # Artifact paths, model IDs, structural constants
|-- weights.yaml                    # Tunable scoring weights and thresholds
|-- validate_submission.py          # CSV format validator
|-- requirements.txt
|-- job_description.txt
|-- candidates.jsonl                # Official dataset, placed locally
|-- team_BuriBuri.csv               # Generated final submission
|-- metadata/
|   |-- JD_contract.yaml            # JD-derived extraction and policy contract
|   `-- validation_set.json
|-- src/
|   |-- jd_intelligence.py          # Phase 0 JD query/config builder
|   |-- features.py                 # Phase 1c feature extraction
|   |-- scorer.py                   # Phase 4 core scoring
|   |-- reranker.py                 # Cross-encoder score merge
|   |-- behavioral.py               # Phase 5 modifiers and ranking
|   |-- explainer.py                # Phase 6 reasoning generation
|   `-- weights.py                  # weights.yaml loader
|-- artifacts/
|   |-- jd_v1_skills.npy
|   |-- jd_hyde_recsys.npy
|   |-- jd_hyde_eval.npy
|   |-- jd_sparse_queries.npz
|   |-- jd_keywords.json
|   |-- jd_config.json
|   |-- faiss_index.bin
|   |-- candidate_ids.json
|   |-- candidate_sparse_matrix.npz
|   |-- candidate_texts.pkl
|   |-- bm25_index.pkl
|   |-- candidate_flags.parquet
|   |-- retrieval_scores_base.parquet
|   |-- retrieval_scores.parquet
|   |-- candidate_features.parquet
|   |-- cross_encoder_scores.parquet
|   `-- run_metadata.json
|-- tests/
`-- docs/
    |-- plan/                       # Architecture explanation by phase
    |-- reference/
    `-- auditfiles/
```

## Key Files

`rank.py`

Final competition entry point. It loads precomputed features, applies scoring, behavioral modifiers, reasoning, and writes the CSV. It does not rebuild artifacts.

`preprocess.py`

Offline artifact builder. It creates JD query vectors, candidate embeddings, FAISS/BM25/sparse artifacts, honeypot flags, retrieval scores, feature parquet, and cross-encoder scores.

`metadata/JD_contract.yaml`

The source of truth for JD-specific skill patterns, location policy, disqualifier terms, and feature contract inputs.

`weights.yaml`

The source of truth for tunable numeric weights and thresholds. Many ranking changes can be made here without preprocessing again.

`constants.py`

The source of truth for file paths, model IDs, and structural constants. Do not hardcode artifact paths in pipeline code.

`artifacts/`

Precomputed files used by `rank.py`. These are already calculated for the current pipeline state. Recalculate them only when the table above says to.

## Output Files

`team_BuriBuri.csv`

Final submission file. It contains:

```text
candidate_id,rank,score,reasoning
```

`artifacts/ranking_debug.csv`

Debug-only output written by `rank.py`. It includes extra fields such as core score, cross-encoder score, final score, reasoning, and concern.

## Submission Compliance

The validator checks:

- exactly 100 data rows
- header is exactly `candidate_id,rank,score,reasoning`
- candidate IDs use the required `CAND_XXXXXXX` format
- ranks 1 through 100 appear exactly once
- scores are non-increasing by rank
- equal-score ties use candidate ID ascending order
- file is UTF-8 CSV

Run:

```bash
python validate_submission.py team_BuriBuri.csv
```

## Demo Sandbox

`app.py` is a lightweight demo path. It is not the final competition ranking path.

Run CLI demo:

```bash
python app.py --candidates ./sample_candidates.json --out output.csv
```

Run local UI:

```bash
python app.py
```

Then open:

```text
http://localhost:7860
```

The demo uses heuristic ranking and does not require the full artifact set.

## Documentation Map

The detailed architecture docs live in `docs/plan/`:

- `docs/plan/common_index.md` - table of contents
- `docs/plan/common_architecture.md` - problem framing, execution model, repository layout
- `docs/plan/phase_0_jd_intelligence.md` - JD contract and BGE-M3 query generation
- `docs/plan/phase_1_corpus_preprocessing.md` - embeddings, FAISS, BM25, honeypots
- `docs/plan/phase_2_multi_signal_retrieval.md` - RRF retrieval and exact recall
- `docs/plan/phase_3_feature_extraction.md` - feature buckets and snippets
- `docs/plan/phase_4_core_scoring.md` - scoring and cross-encoder merge
- `docs/plan/phase_5_behavioral_reranking.md` - behavioral modifiers and penalties
- `docs/plan/phase_6_reason_generation.md` - factual explanation generation
- `docs/plan/phase_7_validation_and_references.md` - validation and references

Use this README for running the project. Use `docs/plan/` when you need to explain the architecture in detail.
