# Judge Guide

This is the fastest path for a reviewer to understand and reproduce the submission.

## Team

Team BuriBuri:

- Sathvik Pilyanam
- Pranathi Mandadi

## What To Run

Windows PowerShell:

```powershell
git lfs install
git lfs pull
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

macOS/Linux:

```bash
git lfs install
git lfs pull
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

For WSL/Ubuntu, use Linux venv commands instead:

```bash
sudo apt update
sudo apt install -y python3-full python3-venv python3-pip git-lfs
git lfs install
git lfs pull
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
```

Use Python 3.11 or 3.12. Python 3.13 is not recommended for this repo because some pinned scientific packages may not have compatible wheels.

Optional tests:

```bash
python -m pytest tests -q
```

## Expected Result

- `team_BuriBuri.csv` with exactly 100 rows.
- Required columns: `candidate_id,rank,score,reasoning`.
- Runtime around 8 seconds locally with current artifacts.
- CPU-only ranking, no network calls, no GPU, no hosted LLM.

Expected console output:

```text
=== Stage B: Ranking ===
  [1/5] Loaded 12,567 candidates
  [2/5] Retrieval scoring -> top 10,000 by RRF
  [3/5] Core + cross-encoder scoring complete (1,000 candidates in pool)
  [4/5] Behavioral modifiers + calibration...
  [5/5] Ranks assigned -> generating reasoning for top 100 candidates

Done in 8.33s  ->  100 candidates ranked

Output files:
  Submission       ./team_BuriBuri.csv
  Ranking trace    artifacts/ranking_debug.csv
  Filtered out     artifacts/hard_disqualified_debug.csv
  Score gaps       artifacts/rank_score_gaps.csv
  Gap diagnostics  artifacts/score_gap_diagnostics.csv
  YOE distribution artifacts/yoe_distribution.csv
  Statistics       docs/ranking_statistics.md
```

## Integrity Declaration

The submitted CSV is generated from the code path in `rank.py`. Ranks come from the internal `true_unclamped_final_score`; submitted scores are sigmoid-mapped display scores derived from that internal score. Ranking uses implemented retrieval, scoring, cross-encoder merge, behavioral modifiers, trust checks, and `weights.yaml` configuration. The reasoning column comes from `src/explainer.py`, using extracted profile facts and evidence snippets.

No candidate was manually inserted, removed, reordered, rescored, or manually given a custom explanation after generation.

## Architecture In One Paragraph

The system precomputes candidate embeddings, sparse vectors, BM25 indexes, honeypot flags, retrieval scores, feature tables, and cross-encoder scores offline. The current CE artifact covers all 12,567 retrieval-pool candidates and was merged from three non-overlapping CE parts, then normalized globally at rank time. At runtime, `rank.py` loads artifacts, filters to the provided candidate IDs, slices the RRF retrieval pool, computes a JD-specific handcrafted score, blends the offline cross-encoder score, runs a lightweight full-profile calibration pass over the final slice, applies behavioral and trust modifiers, assigns deterministic ranks, and generates factual reasoning from extracted snippets.

## Why This Is Not Keyword Stuffing

The JD says the right answer is not to find the most AI keywords. The engine therefore uses:

- retrieval/search and vector evidence
- recommender/ranking systems evidence
- evaluation framework evidence
- product-company and shipper signals
- trust/honeypot checks
- late behavioral reachability modifiers

This lets candidates with real production ranking/recommendation/search experience compete with explicit RAG/vector-search candidates.

## Important Files

| File | Why it matters |
|---|---|
| `README.md` | Main setup and run guide |
| `docs/ARCHITECTURE.md` | Full architecture explanation with diagrams |
| `REPRODUCIBILITY.md` | Exact reproduction and rebuild rules |
| `docs/DATA_AND_ARTIFACTS.md` | Artifact inventory and data boundaries |
| `rank.py` | Final evaluated ranking path |
| `preprocess.py` | Offline artifact builder |
| `weights.yaml` | Scoring and behavior weights |
| `metadata/JD_contract.yaml` | JD-specific extraction contract |
| `src/scorer.py` | Core score formula |
| `src/behavioral.py` | Behavioral modifiers and penalties |
| `src/explainer.py` | Deterministic factual reasoning |

## Manual Review Notes

The `reasoning` column is deterministic and evidence-grounded. It may repeat some structure by design, because reproducibility and factuality are safer than generative prose. Good rows should:

- cite concrete role, company, years, or technical evidence
- connect to retrieval/ranking/recommendation/evaluation JD requirements
- acknowledge logistics or evidence gaps where material
- avoid hallucinated skills, companies, or duration claims
- use stronger tone for top candidates and more cautious tone for lower top-100 candidates

## Current Verification

Latest local checks:

```text
validate_submission.py team_BuriBuri.csv: valid
reasoning: deterministic evidence-grounded rows spot-checked after regeneration
py_compile: rank.py, src/behavioral.py, src/explainer.py pass
rank.py runtime: 8.33 seconds locally
score range: 96.178 to 46.681
```
