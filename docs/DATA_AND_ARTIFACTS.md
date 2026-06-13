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
| `artifacts/cross_encoder_scores.parquet` | `preprocess.py` / `merge_ce.py` | `rank.py` | Offline BGE reranker scores for the full retrieval pool |
| `artifacts/run_metadata.json` | `preprocess.py` | `rank.py` | Candidate count and reference date |
| `artifacts/ranking_debug.csv` | `rank.py` | humans/debug | Debug trace for ranked gate-passing candidates |
| `artifacts/hard_disqualified_debug.csv` | `rank.py` | humans/debug | Candidates assigned near-zero hard-gate scores with reasons |
| `artifacts/rank_score_gaps.csv` | `rank.py` | humans/debug | Adjacent rank score gaps for confidence review |
| `artifacts/score_gap_diagnostics.csv` | `rank.py` | humans/debug | Adjacent-rank true-score gaps with likely causes |
| `artifacts/large_gap_warnings.csv` | `rank.py` | humans/debug | Large Top-40 true-score gaps for manual audit |
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
| `artifacts/candidate_features.parquet` | 0.63 MB |
| `artifacts/retrieval_scores.parquet` | 0.20 MB |
| `artifacts/cross_encoder_scores.parquet` | 0.06 MB |

The current retrieval, feature, and CE artifacts all cover 12,567 candidates. The CE scores were regenerated in three non-overlapping 4,189-row parts and merged into `artifacts/cross_encoder_scores.parquet`; rank-time normalization is global over the merged CE file, not per partition.

After the strict hard-gate update, regenerated ranking artifacts also include `artifacts/hard_disqualified_debug.csv`. The final ranking keeps hard-disqualified candidates in the scored pool with near-zero hard-gate scores rather than manually deleting them; the current official Top 100 contains zero hard-disqualified candidates. Notice and location risks remain in `ranking_debug.csv` as strong penalties/caveats, not hard exclusions.

## Data Governance Notes

- The official Redrob dataset and resource bundle are not owned by Team BuriBuri.
- Do not commit private tokens, hosted API keys, or unrelated personal data.
- Do not manually edit generated submission rows; regenerate from `rank.py`.
- `Resources/` and `candidates.jsonl` may be ignored locally depending on repository packaging.
