# Evidence Rank - Team BuriBuri

Redrob Hackathon submission for the Intelligent Candidate Discovery and Ranking Challenge.

Team BuriBuri: Sathvik Pilyanam, Pranathi Mandadi

This repository ranks the best 100 candidates for the Redrob Senior AI Engineer JD. The system is built as a two-stage retrieval and ranking engine: expensive embedding/index work is precomputed once, while the final `rank.py` submission path is CPU-only, deterministic, and fast.

## Quick Start

```bash
git clone https://github.com/sathvik1610/evidence-rank.git
cd evidence-rank
```

Use Python 3.11 or 3.12. Do not use Python 3.13 for this project because several pinned scientific packages and FAISS/offline dependencies may not have compatible wheels yet.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

WSL/Ubuntu:

```bash
sudo apt update
sudo apt install -y python3-full python3-venv python3-pip git-lfs
git lfs install
git lfs pull

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do not use Windows commands such as `py -3.11` or `.\.venv\Scripts\activate` inside WSL. Use `python3 -m venv .venv` and `source .venv/bin/activate`.

If artifacts are stored through Git LFS, pull them after cloning:

```bash
git lfs install
git lfs pull
```

Place the official candidate file at the repository root:

```text
candidates.jsonl
```

Generate and validate the final submission:

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

Run tests:

```bash
python -m pytest tests -q
```

Only install the full offline/model stack if you need to rebuild preprocessing artifacts:

```bash
pip install -r requirements-offline.txt
```

## Current Submission Metrics

These are local verification metrics for the current generated `team_BuriBuri.csv`.

| Metric | Current value |
|---|---:|
| Output rows | 100 |
| Required columns | `candidate_id,rank,score,reasoning` |
| Runtime for `rank.py` | about 2.1 seconds |
| Ranking constraint | CPU only, no network, no GPU |
| Submission validator | Pass |
| Reasoning factuality audit | 100 rows checked, 0 errors, 0 warnings |
| Full test suite | 94 passed |
| Top score | 95.86758 |
| Rank-100 score | 51.063791 |
| Mean score | 66.1263 |
| Median score | 64.9965 |
| Rank 1 | `CAND_0018499` |
| Rank 10 | `CAND_0061257` |
| Rank 50 | `CAND_0053695` |
| Rank 100 | `CAND_0027801` |

Competition scoring metrics from `Resources/submission_spec.txt`:

| Hidden evaluation metric | Weight |
|---|---:|
| NDCG@10 | 0.50 |
| NDCG@50 | 0.30 |
| MAP | 0.15 |
| P@10 | 0.05 |

Final composite:

```text
0.50 * NDCG@10 + 0.30 * NDCG@50 + 0.15 * MAP + 0.05 * P@10
```

The ranking therefore prioritizes top-10 quality first, then top-50 quality, while still keeping the entire top 100 reasonable and explainable.

## What The System Optimizes For

The JD is not a generic LLM-engineer search. It asks for a founding-team Senior AI Engineer who can own ranking, retrieval, matching, and evaluation systems for a recruiting product.

The engine favors candidates with:

- production retrieval/search/recommendation/ranking ownership
- embeddings, vector DB, hybrid retrieval, BM25, FAISS, Pinecone, OpenSearch, Elasticsearch, or similar systems
- evaluation culture: NDCG, MRR, MAP, Recall@K, A/B testing, offline-to-online calibration, human relevance judgments
- product-company experience, product ML ownership, and shipped systems
- strong Python and hands-on implementation evidence
- reasonable reachability: notice period, response rate, location, relocation, and platform activity

It downranks candidates with:

- pure research without production deployment
- LangChain/OpenAI-wrapper-only experience without deeper ML systems background
- consulting-only careers with weak product ownership
- wrong-domain expertise such as CV/speech/robotics without NLP/IR evidence
- suspicious timelines, skill-duration overclaims, ghost profiles, or impossible profile structure
- low technical depth or weak ranking/evaluation evidence in lower rank bands

## Architecture Summary

The system has two stages.

### Stage A: Offline Preprocessing

Entry point:

```bash
python preprocess.py --candidates ./candidates.jsonl
```

Purpose:

- build JD query artifacts from `job_description.txt` and `metadata/JD_contract.yaml`
- encode candidates with `BAAI/bge-m3`
- build dense FAISS, learned-sparse CSR, and BM25 indexes
- run honeypot, ghost, contradiction, and trust checks directly on JSON fields
- create high-recall retrieval scores using RRF
- extract JD-specific features and evidence snippets
- run `BAAI/bge-reranker-v2-m3` cross-encoder offline
- save all reusable artifacts under `artifacts/`

This stage can take much longer and can use GPU. It is not the final evaluated ranking path.

### Stage B: Runtime Ranking

Entry point:

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
```

Purpose:

- load precomputed features and retrieval scores
- filter to candidate IDs in the input file
- slice the retrieval pool using `weights.yaml`
- compute core score with `src/scorer.py`
- blend precomputed cross-encoder score with handcrafted score
- apply behavioral modifiers and trust penalties with `src/behavioral.py`
- assign deterministic ranks
- generate factual reasoning with `src/explainer.py`
- write `team_BuriBuri.csv` and `artifacts/ranking_debug.csv`

This stage is CPU-only, uses no network calls, and does not load torch, FAISS, FlagEmbedding, or sentence-transformer models.

For the deeper system walkthrough, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Main Methods Used

| Method | Where used | Why it exists |
|---|---|---|
| BGE-M3 dense embeddings | Phase 0/1 preprocessing | Semantic recall for JD/candidate matching |
| BGE-M3 learned sparse vectors | Phase 0/1 preprocessing | Sparse neural lexical matching without runtime model inference |
| FAISS `IndexFlatIP` | Phase 1 preprocessing | Fast dense nearest-neighbor search over normalized candidate vectors |
| BM25 | Phase 1/2 preprocessing | Exact lexical anchor for terms like BM25, FAISS, Pinecone, NDCG, MRR |
| Exact/regex recall lane | Phase 2 preprocessing | Conservative rescue path for high-signal JD terms and career evidence |
| Reciprocal Rank Fusion | Phase 2 preprocessing | Combines dense, sparse, BM25, and exact recall into one high-recall pool |
| Rule-based feature extraction | Phase 3 preprocessing | Converts raw JSON into JD-specific scoring features and factual snippets |
| Honeypot/trust checks | Phase 1f preprocessing | Detects impossible timelines, ghosts, overclaims, and suspicious profiles |
| Handcrafted scoring | Phase 4 runtime | Interpretable JD-specific score based on must-haves, career quality, and product ownership |
| Cross-encoder merge | Phase 4 runtime from offline scores | Improves semantic ordering without runtime model inference |
| Behavioral modifiers | Phase 5 runtime | Applies reachability, notice, location, activity, and trust penalties late |
| Deterministic reasoning | Phase 6 runtime | Generates factual 1-2 sentence explanations without LLM hallucination risk |

## Scoring Weights

Core technical score in `src/scorer.py` is controlled by `weights.yaml`:

| Core bucket | Weight |
|---|---:|
| Must-have evidence | 0.55 |
| Nice-to-have evidence | 0.05 |
| Career quality | 0.15 |
| Product-builder score | 0.25 |

Must-have sub-signals:

| Sub-signal | Weight |
|---|---:|
| Retrieval/search | 0.22 |
| Vector DB / hybrid search | 0.16 |
| Recommendation/ranking systems | 0.20 |
| Evaluation framework | 0.17 |
| Python engineering | 0.05 |

Cross-encoder blend:

| Phase 4 component | Weight |
|---|---:|
| Handcrafted core score | 0.65 |
| Precomputed cross-encoder score | 0.35 |

Behavioral and penalty values are also in `weights.yaml`, including notice period, response rate, location, relocation, ghost/honeypot, social proof, contradiction, and low-density evidence penalties.

## Artifact Guide

`rank.py` expects precomputed artifacts. It does not rebuild missing embeddings or indexes.

Important current artifacts:

| Artifact | Purpose | Approx size |
|---|---|---:|
| `artifacts/candidate_features.parquet` | Feature table consumed by `rank.py` | 0.53 MB |
| `artifacts/retrieval_scores.parquet` | RRF retrieval scores | 0.20 MB |
| `artifacts/cross_encoder_scores.parquet` | Offline cross-encoder scores | 0.11 MB |
| `artifacts/candidate_flags.parquet` | Honeypot, ghost, contradiction, trust flags | 0.84 MB |
| `artifacts/faiss_index.bin` | Dense vector index | 390.63 MB |
| `artifacts/candidate_sparse_matrix.npz` | Learned-sparse candidate matrix | 61.03 MB |
| `artifacts/bm25_index.pkl` | BM25 index | 189.12 MB |
| `artifacts/candidate_texts.pkl` | Serialized candidate text | 187.27 MB |
| `artifacts/candidate_ids.json` | Candidate ID order for indexes | 1.53 MB |

`artifacts/run_metadata.json` currently records:

```text
candidate_count: 100000
reference_date: 2026-05-27
skip_embed: true
```

## When To Recalculate Preprocessing

Most expensive work is already calculated. Use this table before changing code.

| Change made | Required action |
|---|---|
| Only `weights.yaml` changed | Run `rank.py` only |
| `src/scorer.py` changed | Run `rank.py` only |
| `src/behavioral.py` changed | Run `rank.py` only |
| `src/explainer.py` changed | Run `rank.py` only |
| Reasoning wording/templates changed | Run `rank.py` only |
| `src/features.py` changed | Run `preprocess.py --skip-embed`, then `rank.py` |
| Honeypot, ghost, contradiction, exact recall, or feature extraction logic changed in `preprocess.py` | Run `preprocess.py --skip-embed`, then `rank.py` |
| `metadata/JD_contract.yaml` changed for extraction patterns, feature terms, location policy, or exact recall | Run `preprocess.py --skip-embed`, then `rank.py` |
| Candidate dataset changed | Run full `preprocess.py`, then `rank.py` |
| Embedding model, dense/sparse JD query text, FAISS, BM25, sparse matrix, or index construction changed | Run full `preprocess.py`, then `rank.py` |
| Cross-encoder scores need refresh for the current retrieval pool | Run `preprocess.py --only-cross-encoder`, then `rank.py` |

Lightweight preprocessing:

```bash
python preprocess.py --candidates ./candidates.jsonl --skip-embed
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

Full preprocessing:

```bash
python preprocess.py --candidates ./candidates.jsonl
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

Cross-encoder refresh only:

```bash
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

## Validation Checklist

Before submission:

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
python -m pytest tests -q
```

The official CSV must satisfy:

- exactly 100 data rows
- header exactly `candidate_id,rank,score,reasoning`
- each candidate ID exists in the released dataset
- each rank from 1 to 100 appears exactly once
- scores are non-increasing
- score ties use deterministic ordering
- UTF-8 CSV format
- reasoning is factual, JD-connected, honest about concerns, varied enough, and rank-consistent

## Integrity Declaration

`team_BuriBuri.csv` is generated by running `rank.py` over the official candidate file and precomputed artifacts. Candidate ordering, scores, and reasoning are produced by the implemented ranking engine using `weights.yaml`, JD-specific feature extraction, behavioral modifiers, trust checks, and deterministic explanation templates.

Nothing in the CSV is manually ranked, manually reordered, or manually written row by row. The reasoning column is also generated by the engine from extracted profile facts and evidence snippets; it is not produced by a hosted LLM and is not hand-tampered after generation.

## Repository Layout

```text
evidence-rank/
|-- README.md
|-- LICENSE
|-- CONTRIBUTING.md
|-- SECURITY.md
|-- CODE_OF_CONDUCT.md
|-- CHANGELOG.md
|-- CITATION.cff
|-- REPRODUCIBILITY.md
|-- rank.py                         # Final competition ranking entry point
|-- preprocess.py                   # Offline artifact builder
|-- app.py                          # Demo sandbox, not final ranking path
|-- constants.py                    # Artifact paths, model IDs, structural constants
|-- weights.yaml                    # Tunable scoring weights and thresholds
|-- validate_submission.py          # CSV format validator
|-- requirements.txt
|-- requirements-offline.txt
|-- job_description.txt
|-- candidates.jsonl                # Official dataset, placed locally
|-- team_BuriBuri.csv               # Generated final submission
|-- metadata/
|   |-- JD_contract.yaml            # JD-derived extraction and policy contract
|   `-- validation_set.json
|-- src/
|   |-- jd_intelligence.py          # Phase 0 JD query/config builder
|   |-- features.py                 # Phase 3 feature extraction
|   |-- scorer.py                   # Phase 4 core scoring
|   |-- reranker.py                 # Cross-encoder score merge
|   |-- behavioral.py               # Phase 5 modifiers and penalties
|   |-- explainer.py                # Phase 6 reasoning generation
|   `-- weights.py                  # weights.yaml loader
|-- artifacts/                      # Precomputed files used by rank.py
|-- tests/
|-- docs/
|   |-- ARCHITECTURE.md             # Clear architecture guide with Mermaid diagrams
|   |-- plan/                       # Phase-by-phase implementation notes
|   |-- reference/
|   `-- auditfiles/
`-- Resources/                      # Original hackathon resources and specs
```

## Key Files

`rank.py`

Final competition path. Loads precomputed artifacts, scores candidates, applies behavioral modifiers, generates reasoning, writes the CSV, and writes a debug trace.

`preprocess.py`

Offline artifact builder. Creates JD query vectors, candidate embeddings, sparse/BM25/FAISS indexes, flags, retrieval scores, feature parquet, and cross-encoder scores.

Requires the optional offline dependency file:

```bash
pip install -r requirements-offline.txt
```

`metadata/JD_contract.yaml`

Structured JD contract for feature patterns, disqualifiers, location policy, and exact recall terms.

`weights.yaml`

Tunable numeric weights and thresholds. Many ranking changes can be tested by editing this file and rerunning `rank.py`.

`src/explainer.py`

Deterministic reasoning generator. It uses extracted profile facts and snippets only. It does not call an LLM and does not invent skills, companies, or durations.

## Demo Sandbox

`app.py` is a lightweight demo path. It is not the final competition ranking path.

CLI demo:

```bash
python app.py --candidates ./sample_candidates.json --out output.csv
```

Local UI:

```bash
python app.py
```

Open:

```text
http://localhost:7860
```

## Documentation Map

Start with these:

- [docs/JUDGE_GUIDE.md](docs/JUDGE_GUIDE.md) - fastest reviewer path: what to run, what to inspect, and how to understand the submission
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - clear architecture explanation with Mermaid diagrams and stage-by-stage flow
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - exact reproduction commands and when artifacts must be rebuilt
- [docs/DATA_AND_ARTIFACTS.md](docs/DATA_AND_ARTIFACTS.md) - what data/artifacts exist, what generated them, and what `rank.py` consumes

Project governance and metadata:

- [LICENSE](LICENSE) - repository license for original code/docs
- [CONTRIBUTING.md](CONTRIBUTING.md) - development workflow and change checklist
- [SECURITY.md](SECURITY.md) - security and private-reporting policy
- [CHANGELOG.md](CHANGELOG.md) - project change history
- [CITATION.cff](CITATION.cff) - citation metadata

Detailed implementation references:

- [docs/plan/common_index.md](docs/plan/common_index.md) - phase documentation index
- [docs/plan/common_architecture.md](docs/plan/common_architecture.md) - longer implementation notes
- [docs/plan/phase_0_jd_intelligence.md](docs/plan/phase_0_jd_intelligence.md) - JD contract and query generation
- [docs/plan/phase_1_corpus_preprocessing.md](docs/plan/phase_1_corpus_preprocessing.md) - embeddings, FAISS, BM25, honeypots
- [docs/plan/phase_2_multi_signal_retrieval.md](docs/plan/phase_2_multi_signal_retrieval.md) - RRF retrieval and exact recall
- [docs/plan/phase_3_feature_extraction.md](docs/plan/phase_3_feature_extraction.md) - feature buckets and snippets
- [docs/plan/phase_4_core_scoring.md](docs/plan/phase_4_core_scoring.md) - scoring and cross-encoder merge
- [docs/plan/phase_5_behavioral_reranking.md](docs/plan/phase_5_behavioral_reranking.md) - behavioral modifiers
- [docs/plan/phase_6_reason_generation.md](docs/plan/phase_6_reason_generation.md) - factual reasoning generation
- [docs/plan/phase_7_validation_and_references.md](docs/plan/phase_7_validation_and_references.md) - validation and references

Audit/reference notes under `docs/auditfiles/` and `docs/reference/` are supporting history, not required reading for judges.

Use this README to run and reproduce the project. Use `docs/ARCHITECTURE.md` to explain the system in interviews or manual review.
