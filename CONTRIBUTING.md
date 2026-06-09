# Contributing

This repository is a hackathon submission, so changes should preserve reproducibility first.

## Development Setup

```bash
git clone https://github.com/sathvik1610/evidence-rank.git
cd evidence-rank
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use Python 3.11 or 3.12. Avoid Python 3.13 because the pinned scientific/runtime stack may not resolve cleanly there.

Place the official dataset at the repository root:

```text
candidates.jsonl
```

## Main Validation Commands

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv
python validate_submission.py team_BuriBuri.csv
python -m pytest tests -q
```

## Change Rules

- For `weights.yaml`, `src/scorer.py`, `src/behavioral.py`, or `src/explainer.py` changes, rerun `rank.py`.
- For `src/features.py`, JD extraction terms, honeypot logic, exact recall, or contradiction checks, rerun `preprocess.py --skip-embed` before `rank.py`.
- For candidate data, embedding model, FAISS/BM25/sparse index construction, or dense/sparse query changes, rerun full `preprocess.py`.
- For full preprocessing or cross-encoder rebuilds, install `requirements-offline.txt`.
- Do not add hosted API calls to `rank.py`.
- Do not add runtime GPU/model dependencies to `rank.py`.
- Do not manually edit `team_BuriBuri.csv`; regenerate it from `rank.py`.

## Pull Request Checklist

- The final CSV is still valid.
- Candidate order and scores changed only if the change intentionally affects ranking.
- Reasoning remains factual and grounded in candidate JSON/features.
- Tests pass.
- README/docs are updated when commands, artifacts, weights, or architecture change.

## Security Reports

Please do not open a public issue for private credentials, accidental data exposure, or security-sensitive problems. See [SECURITY.md](SECURITY.md).
