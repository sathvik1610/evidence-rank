## 11. Phase 7 — Manual Validation + Weight Tuning

### 11.1 Top-20 manual review

Read each of the top 20 profiles manually. Ask: would a senior recruiter agree this person fits a Senior AI Engineer role at a Series A? Flag:
- Keyword stuffers (lots of skills, no deployment evidence)
- Consultants who somehow ranked high despite the multiplier
- Wrong-domain candidates (CV/speech specialists)

Adjust weights in Phase 4 formula if patterns emerge.

### 11.2 Obvious-fit audit

Manually search the corpus for candidates who fit this description:
- Current title includes "ML Engineer," "AI Engineer," "Search Engineer," or "Relevance Engineer"
- Skills include FAISS, BGE, embedding, or NDCG
- Career history shows a product company

If these candidates are not in the top 10, trace which phase dropped them. The JD says "we'd rather see 10 great matches than 1000 maybes" and NDCG@10 is 50% of the score. A single misplaced obvious-fit in rank 15 instead of rank 5 costs significantly.

### 11.3 Honeypot audit

Scan top 100 for candidates with impossible timelines. If any known honeypot is in the top 10, the dense retrieval is over-indexing on keywords. Lower the dense weight in RRF or strengthen the honeypot gate.

The submission spec requires honeypot rate < 10% in top 100. Verify this before submitting.

### 11.4 Reasoning quality audit

Sample 10 rows from across ranks (not just top). Check:
- Every claim exists in the candidate's profile (no hallucination)
- Snippets reference specific tech names, companies, or numbers
- Tone is rank-appropriate (rank 90 should not have glowing reasoning)
- No two reasoning strings are identical

### 11.5 Pre-submission checklist

**File format:**
- [ ] Submission filename is `team_BuriBuri.csv` (registered team participant ID + `.csv` extension per spec)
- [ ] Encoding is UTF-8
- [ ] Exactly 100 rows of data plus 1 header row
- [ ] Column order is exactly: `candidate_id,rank,score,reasoning`
- [ ] Ranks 1–100 each appear exactly once
- [ ] Score non-increasing from rank 1 to rank 100 (ties allowed; ties broken by `candidate_id` ascending)
- [ ] Scores are not all identical (system is differentiating)
- [ ] No duplicate `candidate_id`s
- [ ] All `candidate_id`s match `^CAND_[0-9]{7}$` and exist in `candidates.jsonl`

**Quality:**
- [ ] Verify score is non-increasing from rank 1 to rank 100 and no duplicate IDs exist
- [ ] Honeypot rate in top 100 < 10% (spec §7: auto-disqualified at Stage 3 if exceeded)
- [ ] Top-20 manual review passed
- [ ] Obvious-fit candidates verified in top 10 (NDCG@10 = 50% of score)
- [ ] **Top 5 are pristine** — spec tiebreak §4: P@5 is the first tiebreaker between equal composites
- [ ] Reasoning variation check: 10 sampled rows are specific, honest, and varied
- [ ] No reasoning hallucinations (every claim exists in the candidate's profile)
- [ ] Rank-appropriate tone: rank 5 is not critical, rank 95 is not glowing

**Repo & sandbox:**
- [ ] Sandbox (HF Spaces / Gradio) accepts ≤100 candidates and ranks them end-to-end under 5 min on CPU
- [ ] `submission_metadata.yaml` complete at repo root
- [ ] `README.md` has exact reproduce command: `python rank.py --candidates ./candidates.jsonl --out ./team_BuriBuri.csv`
- [ ] Git history shows real iteration (at least 8 commits across phases; no single "Add everything" commit)

**Portal metadata (ready before upload — spec §10.2):**
- [ ] Team name: `BuriBuri`
- [ ] Primary contact name
- [ ] Primary contact email
- [ ] Primary contact phone
- [ ] GitHub repository URL (format: `https://github.com/USERNAME/REPO`)
- [ ] Sandbox / demo link (working HF Spaces URL)
- [ ] AI tools declared (multi-select: Claude / ChatGPT / Copilot / Cursor / Gemini / Other / None)
- [ ] Compute environment summary (one line: e.g. `Windows 11, Intel i7, 16GB RAM, Python 3.11`)
- [ ] Team member list (name + email for each member)
- [ ] Methodology summary (≤200 words — strongly recommended for Stage 4)

### 11.6 Dynamic Weight Tuning & Config Injection (weights.yaml)

To prevent code modification during final validation and parameter tuning, the implementation uses **Option A (Dynamic Configuration Injection)**:
- **Conceptual Simplicity in Plan:** The python code snippets within this `plan.md` specify hardcoded default numerical values (e.g., `0.55 * must_have_score` or `multiplier *= 0.80`) to remain clean, self-contained, and highly readable.
- **Dynamic Override in Execution:** In the actual codebase, all scoring weights, multipliers, boosts, and thresholds are loaded dynamically from `weights.yaml` (defined in detail at the bottom of [variables.md](file:///d:/GitHub/evidence-rank/variables.md)) into a central runtime configuration dictionary.
- **No-Code Iteration:** During manual review and validation (Phases 7.1–7.4), tuning parameter thresholds or weight distributions only requires editing `weights.yaml` and re-running the scripts. No Python files need to be opened or modified.

---

## 12. Data Field Reference

| Feature | JSON Path | Type | Notes |
|---|---|---|---|
| years_of_experience | `profile.years_of_experience` | float | 0–50 |
| current_title | `profile.current_title` | string | |
| headline | `profile.headline` | string | |
| summary | `profile.summary` | string | |
| current_company | `profile.current_company` | string | |
| current_industry | `profile.current_industry` | string | |
| location | `profile.location` | string | City, region |
| country | `profile.country` | string | |
| skill name | `skills[i].name` | string | |
| skill proficiency | `skills[i].proficiency` | string | beginner/intermediate/advanced/expert |
| skill endorsements | `skills[i].endorsements` | int | ≥0 |
| skill duration_months | `skills[i].duration_months` | int | ≥0 |
| career title | `career_history[i].title` | string | |
| career description | `career_history[i].description` | string | Primary evidence source |
| career company | `career_history[i].company` | string | |
| career industry | `career_history[i].industry` | string | |
| career duration | `career_history[i].duration_months` | int | |
| education tier | `education[i].tier` | string | tier_1 to tier_4 |
| assessment scores | `redrob_signals.skill_assessment_scores` | dict | skill_name → 0–100 |
| last active | `redrob_signals.last_active_date` | date | YYYY-MM-DD |
| open to work | `redrob_signals.open_to_work_flag` | bool | |
| profile views | `redrob_signals.profile_views_received_30d` | int | ≥0 |
| applications submitted | `redrob_signals.applications_submitted_30d` | int | ≥0 |
| response rate | `redrob_signals.recruiter_response_rate` | float | 0.0–1.0 |
| avg response time | `redrob_signals.avg_response_time_hours` | float | ≥0 |
| notice period | `redrob_signals.notice_period_days` | int | 0–180 |
| github score | `redrob_signals.github_activity_score` | float | -1 (no GitHub) to 100 |
| saved by recruiters | `redrob_signals.saved_by_recruiters_30d` | int | ≥0 |
| interview completion | `redrob_signals.interview_completion_rate` | float | 0.0–1.0 |
| offer acceptance | `redrob_signals.offer_acceptance_rate` | float | -1 (no history) to 1.0 |
| willing to relocate | `redrob_signals.willing_to_relocate` | bool | |
| preferred work mode | `redrob_signals.preferred_work_mode` | string | remote/hybrid/onsite/flexible |
| profile completeness | `redrob_signals.profile_completeness_score` | float | 0–100 |
| endorsements received | `redrob_signals.endorsements_received` | int | ≥0 |
| verified email | `redrob_signals.verified_email` | bool | |
| verified phone | `redrob_signals.verified_phone` | bool | |
| linkedin connected | `redrob_signals.linkedin_connected` | bool | |

---

## 13. Model & Library Choices

| Component | Choice | Size | Notes |
|---|---|---|---|
| Dense embedding | `BAAI/bge-m3` | 570 MB | Dense + sparse in one model. CPU ~2ms/doc. Offline only. |
| BM25 | `rank_bm25` | — | Pure Python. Built at precompute time, pickled to disk. Offline only. |
| Vector index | `faiss-cpu` `IndexFlatIP` | ~300 MB | Exact search on 100K. Built offline; not loaded at rank time. |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | 130 MB | Applied offline to the configured retrieval pool, currently 15,000 candidates. Scores saved to parquet. Offline only. |
| Feature extraction | `regex` + phrase dictionaries | — | Pure Python pattern matching. No NLP library dependency. Both offline and rank time. |
| Reason generation | Pure Python | — | String templates + evidence dict loaded from parquet. Rank time only. |

---

## 14. Dependency Declarations

### 14.1 Offline Dependencies (`preprocess.py` only)

These packages are required to build the precomputed artifacts. They are **not** imported by `rank.py` and do not need to be available at rank time.

```
flagembedding==1.2.5          # BGE-M3 embedder + bge-reranker-v2-m3
faiss-cpu==1.8.0              # FAISS IndexFlatIP build
torch>=2.0.0                  # Required by FlagEmbedding
sentence-transformers>=2.6.0  # Required by FlagEmbedding
```

### 14.2 Runtime Dependencies (`rank.py` only)

These are the only packages that `rank.py` imports. Total cold-import time should be under 2 seconds.

```
pandas==2.2.2
numpy==1.26.4
pyarrow==16.1.0
scipy==1.13.0
rank-bm25==0.2.2              # Required by app.py sandbox BM25 fallback path
```

### 14.3 Demo Dependencies (`app.py` only)

```
gradio==4.36.1
```

Total installed dependency size (offline + runtime + model artifacts) must stay within the 5 GB disk constraint.

> **No SpaCy.** Feature extraction uses pure Python `re` (regex) and phrase dictionaries only. SpaCy is not a dependency at any stage.

---

## 15. Reproducibility Flow

```
preprocess.py  (run once offline, no time limit)
    │
    ├── Phase 0: JD parse → artifacts/jd_config.json / jd_*.npy / jd_sparse_queries.npz
    ├── Phase 1: Embed 100K candidates → artifacts/faiss_index.bin
    │                                 → artifacts/bm25_index.pkl
    │                                 → artifacts/candidate_ids.json
    │                                 → artifacts/candidate_flags.parquet
    │                                 → artifacts/run_metadata.json  (reference_date = max last_active_date)
    ├── Phase 1d: 6-way RRF retrieval (FAISS + sparse + BM25 + exact recall) → artifacts/retrieval_scores.parquet  [top 15,000]
    ├── Phase 1c: Feature extraction (Bucket A/B/C on widened pool) → artifacts/candidate_features.parquet
    │            Preliminary core score computed here to rank candidates for CE selection
    └── Phase 1e: Cross-encoder configured pool (currently 15,000) → artifacts/cross_encoder_scores.parquet

rank.py  (evaluation machine, ≤ 5 minutes wall clock)
    │
    ├── Load artifacts/retrieval_scores.parquet      → top 15,000 candidates (sliced to runtime_top_k)
    ├── Load artifacts/candidate_features.parquet    → Bucket A/B/C ready
    ├── Load artifacts/cross_encoder_scores.parquet  → CE scores ready for configured pool
    ├── Load artifacts/run_metadata.json             → reference_date
    │   reference_date = stored reference_date, or 2026-06-01 fallback
    ├── Phase 4: Weighted score + CE merge → configurable Phase 5 pool
    ├── Phase 5: Behavioral multipliers → top 100
    └── Phase 6: Reason generation → team_BuriBuri.csv
```

**Dataset assumption:** `rank.py` expects precomputed artifacts for the official competition dataset. If `artifacts/candidate_features.parquet` is not found, it exits and asks for preprocessing. If `artifacts/retrieval_scores.parquet` is missing, sample-style runs can still score the available precomputed features, but the full competition path should have retrieval scores present. Runtime embedding generation is not supported.

**Reference date guard:** `rank.py` reads `reference_date` from `artifacts/run_metadata.json` when present and falls back to `2026-06-01` otherwise. It does not recompute embeddings or rebuild date metadata at runtime.

---

### 15.1 Runtime Logging

The current `rank.py` logs the major execution checkpoints to stdout:
- number of feature rows loaded after filtering to the input candidate IDs
- number of rows retained after the Phase 2 RRF cutoff
- configurable Phase 5 candidate-pool size
- Phase 5 and Phase 6 start messages
- output CSV path
- debug trace path
- total runtime in seconds

It does not currently log per-phase timings or peak RSS memory. Add those only if needed for a stricter profiling report.

---

### 15.2 Run Metadata Schema

`artifacts/run_metadata.json` is a simple dictionary generated by `preprocess.py`:

```json
{
  "reference_date": "2026-06-05",
  "run_time": "2026-06-05T12:34:56.000000Z",
  "candidate_count": 100000,
  "skip_embed": false
}
```
`rank.py` uses `reference_date` to compute days inactive.

---

*End of specification. Version 3.3.0.*
