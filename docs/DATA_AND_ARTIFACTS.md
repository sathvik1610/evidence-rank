# Data And Artifacts

This project separates official competition data, generated artifacts, and final outputs.

## Official Inputs

The official dataset should be placed at the repository root:

```text
candidates.jsonl
```

The job description is stored as:

```text
job_description.txt
```

The structured JD contract used by the extractor is:

```text
metadata/JD_contract.yaml
```

For the released dataset summary, trap assumptions, duplicate-description audit counts, behavioral-twin interpretation, and preprocessing reliability notes, see:

```text
docs/DATASET_ANALYSIS.md
```

## Generated Submission

The final generated CSV is:

```text
team_BuriBuri.csv
```

Required columns:

```text
candidate_id,rank,score,reasoning
```

Validate with:

```bash
python validate_submission.py team_BuriBuri.csv
```

## Artifact Inventory

To reproduce the final CSV from existing artifacts, install only:

```bash
pip install -r requirements.txt
```

To rebuild model/index artifacts, install:

```bash
pip install -r requirements-offline.txt
```

| File | Built by | Used by | Purpose |
|---|---|---|---|
| `artifacts/jd_config.json` | `preprocess.py` | preprocessing/debug | Structured JD rule output |
| `artifacts/jd_keywords.json` | `preprocess.py` | retrieval/debug | BM25 keyword anchors |
| `artifacts/jd_v1_skills.npy` | `preprocess.py` | retrieval | Dense JD skills query |
| `artifacts/jd_hyde_recsys.npy` | `preprocess.py` | retrieval | Dense recommender/ranking query |
| `artifacts/jd_hyde_eval.npy` | `preprocess.py` | retrieval | Dense evaluation query |
| `artifacts/jd_sparse_queries.npz` | `preprocess.py` | retrieval | Learned-sparse JD query matrix |
| `artifacts/faiss_index.bin` | `preprocess.py` | preprocessing retrieval | Dense candidate index |
| `artifacts/candidate_sparse_matrix.npz` | `preprocess.py` | preprocessing retrieval | Learned-sparse candidate matrix |
| `artifacts/bm25_index.pkl` | `preprocess.py` | preprocessing retrieval | BM25 index |
| `artifacts/candidate_texts.pkl` | `preprocess.py` | preprocessing retrieval | Serialized candidate texts |
| `artifacts/candidate_ids.json` | `preprocess.py` | preprocessing retrieval | Candidate ID order for indexes |
| `artifacts/candidate_flags.parquet` | `preprocess.py` | `rank.py` through features | Honeypot, ghost, contradiction, trust flags |
| `artifacts/retrieval_scores.parquet` | `preprocess.py` | `rank.py` | RRF retrieval scores |
| `artifacts/candidate_features.parquet` | `preprocess.py` | `rank.py` | JD features, snippets, behavior/profile facts |
| `artifacts/cross_encoder_scores.parquet` | `preprocess.py` | `rank.py` | Offline BGE reranker scores |
| `artifacts/run_metadata.json` | `preprocess.py` | `rank.py` | Candidate count and reference date |
| `artifacts/ranking_debug.csv` | `rank.py` | humans/debug | Debug trace for ranked gate-passing candidates |
| `artifacts/hard_disqualified_debug.csv` | `rank.py` | humans/debug | Candidates removed before final ranking by JD hard gates |
| `artifacts/rank_score_gaps.csv` | `rank.py` | humans/debug | Adjacent rank score gaps for confidence review |
| `artifacts/yoe_distribution.csv` | `rank.py` | humans/debug | YoE diagnostics for Top 10/25/100 |

## Current Artifact Notes

Current `run_metadata.json` records:

```text
candidate_count: 100000
reference_date: 2026-05-27
skip_embed: true
```

Current important artifact sizes:

| File | Approx size |
|---|---:|
| `artifacts/faiss_index.bin` | 390.63 MB |
| `artifacts/bm25_index.pkl` | 189.12 MB |
| `artifacts/candidate_texts.pkl` | 187.27 MB |
| `artifacts/candidate_sparse_matrix.npz` | 61.03 MB |
| `artifacts/candidate_ids.json` | 1.53 MB |
| `artifacts/candidate_flags.parquet` | 0.84 MB |
| `artifacts/candidate_features.parquet` | 0.53 MB |
| `artifacts/retrieval_scores.parquet` | 0.20 MB |
| `artifacts/cross_encoder_scores.parquet` | 0.11 MB |

After the strict hard-gate update, regenerated ranking artifacts should also include `artifacts/hard_disqualified_debug.csv`. The official `team_BuriBuri.csv` should contain only candidates that pass the JD hard gates: no missing true must-have bucket without raw-evidence override and no explicit JD disqualifier flag. Notice and location risks remain in `ranking_debug.csv` as strong penalties/caveats, not hard exclusions.

## Data Governance Notes

- The official Redrob dataset and resource bundle are not owned by Team BuriBuri.
- Do not commit private tokens, hosted API keys, or unrelated personal data.
- Do not manually edit generated submission rows; regenerate from `rank.py`.
- `Resources/` and `candidates.jsonl` may be ignored locally depending on repository packaging.
