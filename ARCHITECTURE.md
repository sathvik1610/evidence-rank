# Evidence Rank Architecture

The canonical architecture document is:

```text
docs/ARCHITECTURE.md
```

This root-level pointer exists so reviewers who open `ARCHITECTURE.md` from the repository root land on the current documentation instead of an older duplicate.

Current runtime summary:

- final entry point: `python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv`
- CPU-only, no network, no hosted LLM calls during ranking
- offline artifacts provide retrieval scores, features, and cross-encoder scores
- current CE coverage is full for the 12,567-candidate retrieval pool; the CE artifact was merged from three non-overlapping parts and normalized globally at rank time
- runtime ranking uses 68% handcrafted score and 32% precomputed cross-encoder score
- a lightweight full-profile calibration pass checks recent/current JD-fit and services-context signals on the final slice
- behavioral modifiers account for response rate, notice period, open-to-work, location, relocation, trust, and honeypot risk
- strict JD disqualifiers and missing must-have profiles receive near-zero hard-gate scores and fall naturally rather than being manually removed
- submitted scores are fixed-scale monotonic display scores derived from internal `true_unclamped_final_score`
- deterministic reasoning is generated from candidate facts and extracted evidence snippets
