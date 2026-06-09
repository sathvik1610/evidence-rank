# Changelog

## 2026-06-09

- Rewrote `README.md` as the main reproduction and implementation guide.
- Added `docs/ARCHITECTURE.md` with stage-by-stage Mermaid diagrams and current metrics.
- Updated plan docs to match the final no-runtime-LLM architecture.
- Polished deterministic reasoning style while keeping ranks and scores frozen.
- Verified final submission format, reasoning factuality, and full tests.

## Earlier Development

- Built two-stage architecture: offline preprocessing plus fast runtime ranking.
- Added BGE-M3 dense/sparse retrieval, BM25, exact recall, RRF, feature extraction, cross-encoder merge, behavioral modifiers, honeypot checks, and deterministic reasoning.
- Iteratively audited top-100 candidates, logistics penalties, evaluation-signal extraction, and reasoning quality.
