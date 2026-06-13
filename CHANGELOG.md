# Changelog

## 2026-06-14

- Cleaned `rank.py` console output: replaced verbose phase/debug prints with 5 numbered progress steps and a grouped file-output summary at the end.
- Removed "Large Top-40 gap warnings" count from console output (files still written to `artifacts/`).
- Score display formula updated to sigmoid mapping onto 42–98 range.
- Updated README, JUDGE_GUIDE, and CHANGELOG to reflect current runtime (~7s), score range (96.178–46.455), actual console output format, and correct score formula.

## 2026-06-13

- Regenerated `team_BuriBuri.csv` from the current `rank.py` path and validated it with the bundled submission validator.
- Updated displayed submission scores to a fixed monotonic calibration of internal `true_unclamped_final_score`.
- Reduced social-proof and reachable-elite score cliffs in `weights.yaml`.
- Improved Stage 4 reasoning quality with stronger JD-specific support, visible manual-review caveats, and cleaned evidence snippets.
- Added score-gap diagnostics: `artifacts/score_gap_diagnostics.csv` and `artifacts/large_gap_warnings.csv`.
- Verified the current cross-encoder artifact covers all 12,567 retrieval-pool candidates after merging three non-overlapping CE parts.
- Refreshed README, architecture, reproducibility, data/artifact, judge-guide, metadata, and phase docs to match the current pipeline.

## 2026-06-09

- Rewrote `README.md` as the main reproduction and implementation guide.
- Added `docs/ARCHITECTURE.md` with stage-by-stage Mermaid diagrams and current metrics.
- Updated plan docs to match the final no-runtime-LLM architecture.
- Polished deterministic reasoning style.
- Verified final submission format, reasoning factuality, and full tests.

## Earlier Development

- Built two-stage architecture: offline preprocessing plus fast runtime ranking.
- Added BGE-M3 dense/sparse retrieval, BM25, exact recall, RRF, feature extraction, cross-encoder merge, behavioral modifiers, honeypot checks, and deterministic reasoning.
- Iteratively audited top-100 candidates, logistics penalties, evaluation-signal extraction, and reasoning quality.
