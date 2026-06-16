# Evidence Rank - Team BuriBuri

Redrob Hackathon submission for the Intelligent Candidate Discovery and Ranking Challenge.

Team BuriBuri: Sathvik Pilyanam, Pranathi Mandadi

**[🚀 Try the Sandbox Demo on Google Colab](https://colab.research.google.com/drive/1VOb_D3VliBd3zvcNntn865UaTFod9xfP?usp=sharing)**
*(Instantly evaluates candidate profiles in your browser without requiring the massive ML artifacts)*

This repository ranks the best 100 candidates for the Redrob Senior AI Engineer JD. The system is built as a two-stage retrieval and ranking engine: expensive embedding/index work is precomputed once, while the final `rank.py` submission path is CPU-only, deterministic, and fast.



## Table of Contents

- [Quick Start](#quick-start)
- [Key Dataset Assumptions](#key-dataset-assumptions)
- [Current Submission Metrics](#current-submission-metrics)
- [Preprocessing Reliability](#preprocessing-reliability)
- [What The System Optimizes For](#what-the-system-optimizes-for)
- [Architecture Summary](#architecture-summary)
  - [Stage A: Offline Preprocessing](#stage-a-offline-preprocessing)
  - [Stage B: Runtime Ranking](#stage-b-runtime-ranking)
- [Key Files Explained](#key-files-explained)
- [Main Methods Used](#main-methods-used)
- [Scoring Weights](#scoring-weights)
- [Artifact Guide](#artifact-guide)
- [When To Recalculate Preprocessing](#when-to-recalculate-preprocessing)
- [Validation Checklist](#validation-checklist)
- [Integrity Declaration](#integrity-declaration)
- [Repository Layout](#repository-layout)
- [Key Files](#key-files)
- [Demo Sandbox](#demo-sandbox)
- [Documentation Map](#documentation-map)

## Quick Start

```bash
git clone https://github.com/sathvik1610/evidence-rank.git
cd evidence-rank
```

Use Python 3.11 or 3.12. Do not use Python 3.13 for this project because several pinned scientific packages and FAISS/offline dependencies may not have compatible wheels yet.

Windows PowerShell:

```powershell
git lfs install
git lfs pull

py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
git lfs install
git lfs pull

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

Download the official candidate dataset and place it at the repository root:

Windows PowerShell:

```powershell
pip install gdown
gdown "https://drive.google.com/uc?id=1DEXK9WEfDAj9hN_IY6FSh_Fu3Il1m0Xw" -O candidates.jsonl
```

macOS/Linux/WSL:

```bash
pip install gdown
gdown "https://drive.google.com/uc?id=1DEXK9WEfDAj9hN_IY6FSh_Fu3Il1m0Xw" -O candidates.jsonl
```

Generate and validate the final submission:

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

Expected console output:

```text
=== Stage B: Ranking ===
  [1/5] Loaded 12,567 candidates
  [2/5] Retrieval scoring -> top 10,000 by RRF
  [3/5] Core + cross-encoder scoring complete (1,000 candidates in pool)
  [4/5] Behavioral modifiers + calibration...
  [5/5] Ranks assigned -> generating reasoning for top 100 candidates

Done in ~7s  ->  100 candidates ranked

Output files:
  Submission       ./team_BuriBuri.csv
  Ranking trace    artifacts/ranking_debug.csv
  Filtered out     artifacts/hard_disqualified_debug.csv
  Score gaps       artifacts/rank_score_gaps.csv
  Gap diagnostics  artifacts/score_gap_diagnostics.csv
  YOE distribution artifacts/yoe_distribution.csv
  Statistics       docs/ranking_statistics.md
```

*(Dynamic ranking statistics are also written to [docs/ranking_statistics.md](docs/ranking_statistics.md) each run)*

Run tests:

```bash
python -m pytest tests -q
```

Only install the full offline/model stack if you need to rebuild preprocessing artifacts:

```bash
pip install -r requirements-offline.txt
```

## Key Dataset Assumptions

The official bundle warns about traps such as keyword stuffers, plain-language Tier 5s, behavioral twins, and a small number of honeypots, but it does not define every trap as an automatic elimination rule. During local corpus audits, exact long role descriptions repeated across two or more companies appeared in a large share of the synthetic candidate pool. Because this pattern is too common to treat as proof of fraud, the ranker does not reject candidates solely for duplicate role text.

Instead, exact repeated role descriptions are treated as repeated evidence. The first occurrence can establish retrieval, ranking, evaluation, or production-system relevance; repeated copies are discounted for career-depth, role-count, and density signals so copied text cannot multiply proof of sustained experience. Independent structure is still preserved: company names, tenure, current role, recency, title/seniority, industry, company size, and Redrob behavioral signals remain available to the ranking logic.

Skill duration metadata is also treated as lower-trust than demonstrated career work. Minor duration inconsistencies reduce confidence where material, but they do not by themselves make a candidate a honeypot unless the profile has a clear impossible contradiction.

For the Python must-have, the runtime gate prefers explicit `Python`/Python-library evidence, but it also accepts narrow Python-native ML tooling proxies such as PyTorch, scikit-learn, MLflow, Kubeflow, or FAISS in a hands-on ML/search engineering context. This avoids excluding senior production ML engineers who omit the literal word `Python`, while still preventing generic vector-database mentions from satisfying the coding requirement.

Career-density scoring follows the JD wording rather than only one canonical phrase list. Singular and compound evidence such as `embedding-based search`, `embedding ranker`, and `ranker variants` is treated as retrieval/ranking evidence because the JD explicitly asks for embeddings, retrieval, ranking, and a working ranker. Generic `A/B testing` remains guarded: it contributes to evaluation density only when the same role already shows search, retrieval, ranking, recommendation, or matching work.

**Keyword Stuffer Mitigation**: To naturally filter out candidates who stuff keywords but lack genuine semantic alignment (trap profiles), the ranking engine calculates a "Cross-Encoder Disagreement Penalty" (`ce_core_delta`). If a candidate's handcrafted regex core score vastly exceeds their AI semantic cross-encoder score (by a threshold of `38.0` points), their score is aggressively penalized via a harsh multiplier. This drops false-positives completely out of the Top 10, ensuring high precision in the final submission.

For the dataset summary, corpus audit counts, and the exact assumptions used for duplicate descriptions, behavioral twins, and skill-duration metadata, see [docs/DATASET_ANALYSIS.md](docs/DATASET_ANALYSIS.md).


## Current Submission Metrics

These are local verification metrics for the current generated `team_BuriBuri.csv`.

| Metric                     | Current value                                                                                  |
| ----------------------------| -----------------------------------------------------------------------------------------------:|
| Output rows                | 100                                                                                            |
| Required columns           | `candidate_id,rank,score,reasoning`                                                            |
| Runtime for `rank.py`      | about 7 seconds locally                                                                        |
| Ranking constraint         | CPU only, no network, no GPU                                                                   |
| Submission validator       | Pass                                                                                           |
| Reasoning factuality audit | Deterministic evidence-grounded reasoning; final rows manually spot-checked after regeneration |
| Full test suite            | See local `pytest` run; `rank.py` and edited modules compile                                   |
| Top score                  | 96.178                                                                                         |
| Rank-100 score             | 46.455                                                                                         |
| Mean score                 | 59.149                                                                                         |
| Median score               | 54.299                                                                                         |
| Rank 1                     | `CAND_0046525`                                                                                 |
| Rank 10                    | `CAND_0088025`                                                                                 |
| Rank 50                    | `CAND_0026532`                                                                                 |
| Rank 100                   | `CAND_0078492`                                                                                 |

## Preprocessing Reliability

The expensive preprocessing stage reduces the official 100,000-candidate pool to the feature pool consumed by `rank.py`. The current feature pool contains 12,567 candidates, built as a union of two complementary recall paths:

- a semantic RRF base from dense BGE-M3 retrieval, learned-sparse BGE-M3 retrieval, and BM25
- a full-corpus exact/regex rescue lane over JD-critical career evidence such as retrieval, search, recommendation, ranking, vector/hybrid search, evaluation, Python, and production shipping language

This is intentionally wider than the final top-100 requirement. The goal is high recall before scoring, not early precision. The exact recall lane scans all 100,000 candidates and the current artifacts include every candidate from that lane's top 10,000. The resulting 12,567-row pool is therefore not just "top embedding matches"; it also protects against missing strong plain-language profiles that describe recommendation, ranking, or evaluation work without fashionable RAG/vector keywords.

Current artifact coverage checks:

| Check | Current result |
|---|---:|
| Current retrieval pool | 12,567 candidates |
| Feature rows | 12,567 candidates |
| Cross-encoder rows | 12,567 candidates |
| Exact recall top-10K missing from feature pool | 0 |
| Final top-100 outside feature pool | 0 |
| Final top-100 with impossible/suspicious/ghost flags | 0 |

The current cross-encoder artifact was regenerated in three non-overlapping parts and merged into `artifacts/cross_encoder_scores.parquet`. Each part contains 4,189 candidates, the merged file contains all 12,567 retrieval-pool candidates, and rank-time normalization is applied globally after merge rather than per part.

The main residual risk is semantic-only candidates ranked outside the older semantic base but not caught by exact recall. In practice that risk is reduced by the full-corpus exact lane, the breadth of JD patterns, and the fact that the audited top-100 candidates came from the semantic base. After the strict hard-gate update, final submission artifacts must be regenerated and revalidated before making a current top-100 claim.

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

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        EVIDENCE RANK — TEAM BURIBURI                            ║
║                    Intelligent Candidate Discovery & Ranking                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  INPUT
  ─────
  candidates.jsonl  (100,000 candidates)
  job_description.txt + metadata/JD_contract.yaml
         │
         │
╔════════▼═══════════════════════════════════════════════════════════════════════╗
║  STAGE A — OFFLINE PREPROCESSING   (preprocess.py)         runs once, ~hours  ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  PHASE 0 · JD Intelligence                                                     ║
║  ┌──────────────────────────────────────────────────────────────────────────┐  ║
║  │  Parse JD contract → must-haves, disqualifiers, location policy,        │  ║
║  │  exact recall terms, CE query profile (synthetic perfect candidate)      │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                              │                                                  ║
║  PHASE 1 · Corpus Encoding                                                      ║
║  ┌──────────────────────────────────────────────────────────────────────────┐  ║
║  │  BGE-M3 dense embeddings  →  FAISS IndexFlatIP          (390 MB)        │  ║
║  │  BGE-M3 learned sparse    →  CSR sparse matrix           (61 MB)        │  ║
║  │  BM25                     →  BM25 index                 (189 MB)        │  ║
║  │  Honeypot / ghost / timeline contradiction checks                        │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                              │                                                  ║
║  PHASE 2 · Multi-Signal Retrieval                                               ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────────────┐  ║
║  │  Dense ANN      │  │  Sparse neural  │  │  BM25 lexical + exact/regex    │  ║
║  │  (semantic)     │  │  (lexical-ish)  │  │  rescue lane (all 100K scanned)│  ║
║  └────────┬────────┘  └────────┬────────┘  └──────────────┬─────────────────┘  ║
║           └───────────────────►│◄─────────────────────────┘                    ║
║                          RRF FUSION                                             ║
║                              │                                                  ║
║                    12,567-candidate pool                                        ║
║                              │                                                  ║
║  PHASE 3 · Feature Extraction                                                   ║
║  ┌──────────────────────────────────────────────────────────────────────────┐  ║
║  │  JD-specific signals per candidate:                                      │  ║
║  │  retrieval_search · vector_db_hybrid · ltr_reranking · eval_framework   │  ║
║  │  python_coding · sys_experience · career_ir_density · product_builder   │  ║
║  │  evidence snippets · must-have gaps · trust/flag scores                  │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                              │                                                  ║
║  PHASE 3b · Cross-Encoder Scoring  (BGE-Reranker-v2-M3, local, no network)     ║
║  ┌──────────────────────────────────────────────────────────────────────────┐  ║
║  │  Each candidate profile  ←→  synthetic perfect-candidate resume         │  ║
║  │  Output: semantic similarity score for all 12,567 candidates             │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                              │                                                  ║
║              artifacts/*.parquet  (< 2 MB total for rank.py)                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                               │
                               │  (precomputed artifacts loaded at startup)
                               │
╔══════════════════════════════▼═════════════════════════════════════════════════╗
║  STAGE B — RUNTIME RANKING   (rank.py)                      ~7 seconds, CPU   ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  [1/5]  Load 12,567 candidates from artifacts                                  ║
║                                                                                ║
║  [2/5]  RRF cutoff  →  top 10,000 by retrieval score                           ║
║                                                                                ║
║  [3/5]  Core scoring  (top 1,000 pool)                                         ║
║  ┌──────────────────────────────────────────────────────────────────────────┐  ║
║  │                                                                          │  ║
║  │   Handcrafted JD score  ──────────────────────────────  68%             │  ║
║  │   ┌──────────────────────────────────────────────────┐                  │  ║
║  │   │  must-have evidence      55%  of core score      │                  │  ║
║  │   │    retrieval / search     22%                     │                  │  ║
║  │   │    vector DB / hybrid     16%                     │                  │  ║
║  │   │    ranking systems        20%                     │                  │  ║
║  │   │    eval framework         17%                     │                  │  ║
║  │   │    python engineering      5%                     │                  │  ║
║  │   │  career quality          15%  of core score       │                  │  ║
║  │   │  product-builder score   25%  of core score       │                  │  ║
║  │   │  nice-to-have             5%  of core score       │                  │  ║
║  │   └──────────────────────────────────────────────────┘                  │  ║
║  │                                                                          │  ║
║  │   Precomputed CE score  ──────────────────────────────  32%             │  ║
║  │                                                                          │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  [4/5]  Behavioral modifiers  +  eval-aware calibration                        ║
║  ┌──────────────────────────────────────────────────────────────────────────┐  ║
║  │  BONUSES                         PENALTIES                               │  ║
║  │  + same-project full-system      × notice period > 90d                  │  ║
║  │  + recruiter workflow match      × outside India / no relocation         │  ║
║  │  + passive + responsive + fit    × low response rate (< 40%)            │  ║
║  │  + split-career core coverage    × partial system + logistics risk       │  ║
║  │                                  × no retrieval/eval evidence            │  ║
║  │                                  × keyword stuffer (CE vs core gap)      │  ║
║  │  HARD DISQUALIFIERS (score → 0)                                          │  ║
║  │  impossible timeline · ghost profile · wrong domain · langchain-only     │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
║  [5/5]  Deterministic ranks assigned  →  factual reasoning generated           ║
║         (evidence snippets only — no LLM, no hallucination)                   ║
║                                                                                ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                               │
  OUTPUT
  ──────
  team_BuriBuri.csv            →  100 ranked candidates
  artifacts/ranking_debug.csv  →  full scoring trace
  docs/ranking_statistics.md   →  score distributions
```

### Stage A: Offline Preprocessing

Entry point:

```bash
python preprocess.py --candidates ./candidates.jsonl
```

This stage can use GPU and runs once. It is not the final evaluated ranking path. Saves all reusable artifacts under `artifacts/`.

### Stage B: Runtime Ranking

Entry point:

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
```

CPU-only, no network calls, no torch/FAISS/FlagEmbedding at runtime.

For the deeper system walkthrough, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Key Files Explained

- **`metadata/JD_contract.yaml`**: The source of truth for all hard logic. Instead of parsing the unstructured `job_description.txt` at runtime, we broke the JD down into strict data structures (must-have skills, preferred cities, minimum notice periods). The Regex pipeline and behavioral gates pull directly from this file to ensure zero hallucination.
- **`ce_query_profile.txt`**: The "perfect candidate" synthetic text. Because Cross-Encoders compare two pieces of text for semantic similarity, we couldn't just feed it a bulleted JD. We wrote an ideal, synthesized resume summary describing a candidate who perfectly fits the Redrob JD. The AI Cross-Encoder scores every candidate by measuring how semantically similar their profile is to this text.

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
| Cross-encoder merge | Phase 4 runtime from offline scores | Improves semantic ordering without runtime model inference; current CE scores cover the full 12,567-candidate retrieval pool |
| Runtime profile calibration | Phase 5 runtime over top slice | Reads full profile text for recent full-plan JD fit and current services context |
| JD hard-gate scoring | Phase 5 runtime | Applies near-zero scores to missing-must-have and true disqualifier profiles so they fall naturally rather than being pre-filtered |
| Behavioral modifiers | Phase 5 runtime | Applies reachability, notice, location, activity, and trust penalties late after technical scoring |
| Deterministic reasoning | Phase 6 runtime | Generates factual 1-2 sentence explanations without LLM hallucination risk |

## Scoring Weights

Core technical score in `src/scorer.py` is controlled by `weights.yaml`:

| Core bucket           | Weight |
| -----------------------| -------:|
| Must-have evidence    | 0.55   |
| Nice-to-have evidence | 0.05   |
| Career quality        | 0.15   |
| Product-builder score | 0.25   |

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
| Handcrafted core score | 0.68 |
| Precomputed cross-encoder score | 0.32 |

Behavioral and penalty values are also in `weights.yaml`, including notice period, response rate, location, relocation, ghost/honeypot, social proof, contradiction, services-context, full-profile calibration, and low-density evidence penalties.

The current JD uses hard-gate scoring for capability and trust fit. A candidate with a true must-have gap or JD-explicit disqualifier receives a near-zero final score and falls out of the Top 100 naturally rather than being manually removed before ranking. Notice period and location/no-relocation are strong hiring-friction penalties, not hard exclusions. Hard-disqualification diagnostics are written to `artifacts/hard_disqualified_debug.csv` during `rank.py`; the current Top 100 contains zero hard-disqualified candidates.

The submitted `score` column is a sigmoid mapping of the internal final score onto a 42-98 display range: `visible_score = 42 + 56 * sigmoid((true_unclamped_final_score - 68) / 12)`. Ranking is assigned from the internal `true_unclamped_final_score`; the displayed score is kept on a human-readable scale and is not normalized relative to rank 100.

## Artifact Guide

`rank.py` expects precomputed artifacts. It does not rebuild missing embeddings or indexes.

Important current artifacts:

| Artifact                                 | Purpose                                     | Approx size |
| ------------------------------------------| ---------------------------------------------| ------------:|
| `artifacts/candidate_features.parquet`   | Feature table consumed by `rank.py`         | 0.63 MB     |
| `artifacts/retrieval_scores.parquet`     | RRF retrieval scores                        | 0.20 MB     |
| `artifacts/cross_encoder_scores.parquet` | Offline cross-encoder scores for full retrieval pool | 0.06 MB     |
| `artifacts/candidate_flags.parquet`      | Honeypot, ghost, contradiction, trust flags | 0.84 MB     |
| `artifacts/faiss_index.bin`              | Dense vector index                          | 390.63 MB   |
| `artifacts/candidate_sparse_matrix.npz`  | Learned-sparse candidate matrix             | 61.03 MB    |
| `artifacts/bm25_index.pkl`               | BM25 index                                  | 189.12 MB   |
| `artifacts/candidate_texts.pkl`          | Serialized candidate text                   | 187.27 MB   |
| `artifacts/candidate_ids.json`           | Candidate ID order for indexes              | 1.53 MB     |
| `artifacts/hard_disqualified_debug.csv`  | JD hard-gate exclusions from final ranking  | generated by `rank.py` |
| `artifacts/score_gap_diagnostics.csv`    | Adjacent-rank true-score gap causes         | generated by `rank.py` |
| `artifacts/large_gap_warnings.csv`       | Top-40 large-gap warnings for audit         | generated by `rank.py` |

`artifacts/run_metadata.json` currently records:

```text
candidate_count: 100000
reference_date: 2026-05-27
skip_embed: true
```

## When To Recalculate Preprocessing

Most expensive work is already precomputed. Use this table to decide what to rerun before touching code.

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
| Cross-encoder scores need refresh for the current retrieval pool | Run `preprocess.py --only-cross-encoder`, or split/merge CE parts, then run `rank.py` |

---

### Ranking only (no preprocessing change)

Use when only `weights.yaml`, scoring, behavioral, explainer, or reasoning template changed:

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

---

### Lightweight preprocessing (`--skip-embed`)

Use when `src/features.py`, `preprocess.py` feature/exact-recall/honeypot logic, or `metadata/JD_contract.yaml` changed — but embeddings and indexes are still current.

No offline GPU dependencies needed for this path.

```bash
python preprocess.py --candidates ./candidates.jsonl --skip-embed
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

---

### Cross-encoder refresh only (`--only-cross-encoder`)

Use when the retrieval pool is current but reranker scores need regeneration. Requires offline dependencies and a GPU session.

```bash
pip install -r requirements-offline.txt
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

---

### Full preprocessing

Use when the candidate dataset, embedding model, dense/sparse JD query text, FAISS, BM25, sparse matrix, or index construction changed. Requires offline dependencies and a GPU session.

> **Note**: A full rerun may produce a slightly different top-100 because the upstream retrieval pool can change. Do not run this as a last-minute step unless there is time to compare and audit the regenerated CSV against the current one.

```bash
pip install -r requirements-offline.txt
python preprocess.py --candidates ./candidates.jsonl
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

---

### Parallel cross-encoder refresh (three GPU sessions)

Use when you want to split CE scoring across multiple GPU machines and merge afterward.

Step 1 — split the retrieval pool:

```bash
pip install -r requirements-offline.txt
python split_retrieval.py --parts 3
```

Step 2 — run CE scoring on each part in separate GPU sessions (one per part):

```bash
# Session 1
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder --retrieval-part artifacts/retrieval_scores_part1.parquet --ce-out artifacts/cross_encoder_scores_part1.parquet

# Session 2
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder --retrieval-part artifacts/retrieval_scores_part2.parquet --ce-out artifacts/cross_encoder_scores_part2.parquet

# Session 3
python preprocess.py --candidates ./candidates.jsonl --only-cross-encoder --retrieval-part artifacts/retrieval_scores_part3.parquet --ce-out artifacts/cross_encoder_scores_part3.parquet
```

Step 3 — merge and rank:

```bash
python merge_ce.py artifacts/cross_encoder_scores_part1.parquet artifacts/cross_encoder_scores_part2.parquet artifacts/cross_encoder_scores_part3.parquet --output artifacts/cross_encoder_scores.parquet
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

> **Important**: Do not normalize CE scores per partition. `merge_ce.py` concatenates raw CE logits and `src/reranker.py` normalizes the merged file globally at rank time.

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
|   |-- runtime_calibration.py      # Full-profile calibration for final runtime slice
|   |-- behavioral.py               # Phase 5 modifiers and penalties
|   |-- explainer.py                # Phase 6 reasoning generation
|   `-- weights.py                  # weights.yaml loader
|-- artifacts/                      # Precomputed files used by rank.py
|-- tests/
|-- docs/
|   |-- ARCHITECTURE.md             # Clear architecture guide with Mermaid diagrams
|   |-- DATASET_ANALYSIS.md         # Dataset traps, assumptions, and corpus audit notes
|   |-- DATA_AND_ARTIFACTS.md       # Artifact inventory and data governance
|   |-- JUDGE_GUIDE.md              # Reviewer quick path
|   |-- plan/                       # Phase-by-phase implementation notes
|   `-- reference/                  # Current concise variable reference
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
- [docs/DATASET_ANALYSIS.md](docs/DATASET_ANALYSIS.md) - released dataset summary, trap assumptions, duplicate-description audit, and preprocessing reliability notes
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

Historical duplicate audit/reference documents have been pruned. Current numeric tuning authority remains `weights.yaml`, `constants.py`, and the implementation; regenerate diagnostics whenever the generated top 100 changes materially.

Use this README to run and reproduce the project. Use `docs/ARCHITECTURE.md` to explain the system in interviews or manual review.
