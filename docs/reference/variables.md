# Evidence-Rank — Variable & Weight Reference
**Version:** post-review (matches plan.md v3.3.0+, weights.yaml config injection confirmed — Option A)
**Purpose:** One-stop tuning reference. Every number in the system lives here with its stage, type, and effect. All weights and thresholds are loaded at runtime from `weights.yaml` — see §Weights Config at the bottom. No Python file needs to be edited to tune the system.

---

## How to Read This File

| Symbol | Meaning |
|--------|---------|
| `×` | Multiplicative — applied as `score × value` |
| `+` | Additive — applied as `score + value` |
| `HARD` | Hard exclusion — score forced to 0.0 |
| `FILTER` | Removes candidate from pool entirely |
| `SCORE` | Contributes to a computed sub-score |

---

## Stage 1 — Hard Exclusions (Offline, Phase 1)

### 1A. Honeypot Detection
> Any single condition true → `is_honeypot = True` → **final_score = 0.0 (HARD)**

| Variable | Source Field | Condition | Effect |
|----------|-------------|-----------|--------|
| Expert skill with zero use | `skills[i].proficiency`, `skills[i].duration_months` | proficiency in {expert, advanced} AND duration == 0 | HARD zero |
| Single role longer than career | `career_history[i].duration_months`, `profile.years_of_experience` | `role_duration > (yoe × 12) + 12` | HARD zero |
| Career length impossible overlap | `career_history[i].duration_months`, `profile.years_of_experience` | sum(role_durations) > `(yoe * 12 * 1.5) + 12` | HARD zero |

---

### 1B. Consistency Signals (Extracted Phase 1, Penalized Phase 5)
> Rather than hard eliminations, these signal a suspicious profile.

| Variable | Source Field | Logic | Output |
|----------|-------------|-----------|-------|
| Skill duration vs career length | `skills[i].duration_months`, `profile.years_of_experience` | count where `skill_duration > (yoe × 12) + 6` | `contradiction_skill_duration` |
| Expert skill with bad assessment | `skills[i].proficiency`, `redrob_signals.skill_assessment_scores` | count where proficiency in {expert, advanced} AND assessment_score < **40** | `contradiction_assessment` |

---

### 1C. Ghost Profile Filter
> **ALL FOUR** conditions must be true simultaneously → `is_ghost = True` → **final_score = 0.0 (HARD)**

| Variable | Source Field | Threshold | Logic |
|----------|-------------|-----------|-------|
| Days inactive | `redrob_signals.last_active_date` | > **365** days | AND |
| Recruiter response rate | `redrob_signals.recruiter_response_rate` | < **0.05** | AND |
| Open to work | `redrob_signals.open_to_work_flag` | == **False** | AND |
| Recent applications | `redrob_signals.applications_submitted_30d` | == **0** | AND |

Expected volume filtered: ~1–3% of corpus.

---

### 1C. Disqualifier Flags
> Boolean tags saved to `candidate_flags.parquet`. Used as multipliers later (not hard exclusions).

| Flag | Source Fields | Condition | Used In |
|------|-------------|-----------|---------|
| `consulting_only` | `career_history[i].company`, `.industry` | `product_ratio == 0.0` (100% consulting) | Career quality ×0.4, Product builder ×0.4 |
| `research_only` | `career_history[i].title` | No engineering/developer titles in entire career | Career quality ×0.5, Product builder ×0.5, Soft penalty ×0.40 |
| `wrong_domain` | skills, career descriptions | CV/speech/robotics terms present AND no NLP/IR terms | Career quality ×0.3, Product builder ×0.3, Soft penalty ×0.50 |
| `product_ratio` | `career_history[i].company`, `.duration_months` | `1.0 - (consulting_months / total_months)` | Phase 4 product_builder_score, Phase 5 90-day alignment |

**Consulting firm list (21 firms):** tcs, infosys, wipro, accenture, cognizant, capgemini, hcl, tech mahindra, mphasis, hexaware, mindtree, ltimindtree, l&t infotech, niit technologies, zensar, mastech, syntel, kpit, cyient, birlasoft, persistent systems

---

## Stage 2 — Retrieval (Offline Phase 1d → Runtime Phase 2)

### 2A. FAISS Dense Retrieval
| Parameter | Value | Tuning Effect |
|-----------|-------|---------------|
| JD vectors | 2 (skills-focused + ideal-candidate) | More vectors = broader recall but more RRF inputs |
| Top-k per vector | **2,000** | ↑ = broader recall, ↓ = faster |
| Model | `BAAI/bge-m3` (570 MB) | Fixed — offline only |

### 2B. BM25 Sparse Retrieval
| Parameter | Value | Tuning Effect |
|-----------|-------|---------------|
| Top-k | **2,000** | ↑ = more keyword-match candidates included |
| Query | 37 JD keywords | Modify `jd_keywords.json` to change |
| Library | `rank_bm25` BM25Okapi | — |

### 2C. Reciprocal Rank Fusion (RRF)
| Parameter | Value | Tuning Effect |
|-----------|-------|---------------|
| k constant | **60** | ↑ = dampens top-rank advantage, ↓ = rewards rank 1 heavily |
| Formula | `Σ 1/(k + rank_i)` | Sums across all 3 retrievers |
| Output pool | **N = 3,000** (target) | Experiment with 2,000–3,000; check recall on validation set |

### 2D. Soft Activity Boost (pre-RRF)
Candidates with `open_to_work == True` OR `last_active < 90d` get rank boosted by **200 positions** within retrieval step only.

---

## Stage 3 — Feature Extraction (Offline Phase 1c → Runtime Phase 3)

### 3A. Bucket A — Evidence Scores (0–3 per domain)

| Score | Condition |
|-------|-----------|
| **0** | Skill not found anywhere |
| **1** | Skill in skills section only (claim, no proof) |
| **2** | Skill found in career description (project-level proof) |
| **3** | Career description + production/scale signals present |

**Production signals detected by:** `PRODUCTION_PATTERNS` (regex) — "deployed", "real users", "latency", "p99", "shipped to production", "million requests", "billion queries", "qps", "inference serving"

**Assessment score boost:** If platform assessment ≥ **70** AND evidence score ≥ 1 → `score = min(score + 0.5, 3.0)`

| Domain | Redrob Signal | Phase 4 Weight | Bucket |
|--------|--------------|----------------|--------|
| `retrieval_search` | FAISS, Pinecone, Elasticsearch, dense retrieval, semantic search | **0.25** (must-have) | A |
| `vector_db_hybrid` | Vector DB, hybrid search, ANN, sparse+dense | **0.20** (must-have) | A |
| `eval_framework` | NDCG, MRR, MAP, A/B testing, offline/online eval | **0.15** (must-have) | A |
| `ltr_reranking` | LambdaMART, learning-to-rank, cross-encoder | **0.07** (nice-to-have) | A |
| `llm_integration` | LLM, LoRA, QLoRA, RAG, fine-tuning | **0.03** (nice-to-have) | A |
| `python_coding` | Python in career descriptions (not just skills section) | — (evidence only, used in explanations) | A |

---

### 3B. Bucket B — Career Quality Signals

| Signal | How Computed | Range | Used In |
|--------|-------------|-------|---------|
| `product_ratio` | `1 - (consulting_months / total_months)` | 0.0–1.0 | product_builder_score (×0.35), 90-day alignment (×0.20) |
| `deploy_signal` | Count of PRODUCTION_PATTERNS in career text / 5.0, capped at 1.0 | 0.0–1.0 | product_builder_score (×0.30) |
| `experience_recency` | Most recent role has retrieval/ranking terms? 1.0 / 0.5 | 0.5 or 1.0 | career_quality (×0.04) |
| `depth_signal` | Roles with retrieval evidence / 2.0, capped at 1.0 | 0.0–1.0 | career_quality (×0.03) |
| `shipper_ratio` | shipper_count / (shipper_count + researcher_count) | 0.0–1.0 | product_builder_score (×0.20) |
| `writing_signal` | avg career description length | see table below | logistical ×mult |
| `sys_experience_score` | Built search/ranking/recsys with production? | 0.0, 0.5, or 1.0 | career_quality (×0.08) |
| `product_builder_score` | Composite (see Phase 4) | 0.0–1.0 | Phase 4 outer weight **0.20** |
| `ownership_signal` | "built from scratch", "co-founder", "end-to-end", etc. | bool | product_builder_score (×0.15) |

**Writing signal thresholds:**
| Avg description length | `writing_signal` | Logistical ×mult |
|------------------------|-----------------|------------------|
| ≥ 150 characters | 1.00 | ×1.00 |
| 60–149 characters | 0.95 | ×0.95 |
| < 60 characters | 0.90 | ×0.90 |

---

### 3C. Bucket C — JD Fit Gaps

| Signal | Source | Condition | Used In |
|--------|--------|-----------|---------|
| `title_velocity_flag` | `career_history[i].duration_months` | avg tenure < **18** months AND ≥ 3 roles | Soft penalty ×0.80 |
| `code_stopped` | `profile.current_title`, `profile.years_of_experience` | title in {architect, vp, director, cto, head of} AND yoe > **8** | Soft penalty ×0.75 |
| `langchain_only_flag` | career text, skill durations | ≥ 2 framework-demo terms AND no pre-LLM production terms AND total AI skill months < **12** | Soft penalty ×0.45 |
| `closed_source_flag` | github_score, career text | yoe ≥ **5** AND no external validation (GitHub/papers/talks) | Soft penalty ×0.80 |
| `external_validation` | `redrob_signals.github_activity_score`, career text | github_score > 0 OR external validation terms in text | Reduces closed_source_flag |
| `seniority_score` | `profile.years_of_experience` | see table below | Logistical ×mult |

**Seniority score table:**
| YoE Range | `seniority_score` | Logistical ×mult |
|-----------|------------------|------------------|
| 5–9 years | **1.00** | ×1.00 (sweet spot) |
| 4–5 years | **0.85** | ×0.85 |
| 10–12 years | **0.90** | ×0.90 |
| 3–4 years | **0.65** | ×0.65 |
| > 12 years | **0.80** | ×0.80 |
| < 3 years | **0.40** | ×0.40 |

---

### 3D. Consistency Score

**Purpose:** Detect keyword stuffers. Checks expert/advanced claims against actual career evidence.

**Formula:**
```
consistency_score = 1.0 - 0.7 × (contradicted_claims / total_expert_claims)
```

| Check | Condition | Penalty |
|-------|-----------|---------|
| Title contradiction | Career titles suggest completely different domain | +1 contradiction |
| Duration too short | Skill duration > 0 AND < **12** months | +1 contradiction |
| No career text evidence | Skill name not found in any career description | +0.5 contradiction |
| Trigger threshold | contradictions ≥ **1** → marks claim as contradicted | — |

| consistency_score | Meaning | Soft penalty effect |
|------------------|---------|---------------------|
| **1.00** | No contradictions | ×1.00 |
| **0.70** | ~43% of expert claims contradicted | ×0.70 |
| **0.30** | All expert claims contradicted (keyword stuffer) | ×0.30 |

Applied in Phase 5 soft_penalties as a direct multiplier.

---

## Stage 4 — Core Scoring (Runtime Phase 4)

### 4A. Core Score Formula (SCORE, final range ≈ 0.0–1.0)

```
core_score = 0.55 × must_have_score
           + 0.10 × nice_to_have_score
           + 0.15 × career_quality_score
           + 0.20 × product_builder_score
```

---

### 4B. Must-Have Score (55% of core_score)

```
must_have_raw = 0.25 × (retrieval_search / 3)
              + 0.20 × (vector_db_hybrid / 3)
              + 0.15 × (eval_framework / 3)

must_have_score = must_have_raw / 0.60   [normalized to 0–1]
```

**Hard cap:** if `retrieval_search == 0 AND vector_db_hybrid == 0 AND sys_experience_score == 0` → `must_have_raw = min(must_have_raw, 0.5)`

| Domain | Inner Weight |
|--------|-------------|
| retrieval_search | **0.25** |
| vector_db_hybrid | **0.20** |
| eval_framework | **0.15** |

---

### 4C. Nice-to-Have Score (10% of core_score)

```
nice_to_have_score = (0.07 × (ltr_reranking / 3) + 0.03 × (llm_integration / 3)) / 0.10
```

| Domain | Inner Weight |
|--------|-------------|
| ltr_reranking | **0.07** |
| llm_integration | **0.03** |

---

### 4D. Career Quality Score (15% of core_score)

```
career_quality_raw = 0.08 × sys_experience_score
                   + 0.04 × experience_recency
                   + 0.03 × depth_signal

[Multipliers applied to raw before normalizing:]
  consulting_only → ×0.4
  research_only   → ×0.5
  wrong_domain    → ×0.3

career_quality_score = career_quality_raw / 0.15   [normalized to 0–1]
```

| Signal | Inner Weight |
|--------|-------------|
| sys_experience_score | **0.08** |
| experience_recency | **0.04** |
| depth_signal | **0.03** |

---

### 4E. Product Builder Score (20% of core_score)

Computed in Bucket B (Phase 3) so career-description evidence is available.

```
product_builder_score = 0.35 × product_ratio
                      + 0.30 × deploy_signal
                      + 0.20 × shipper_ratio
                      + 0.15 × ownership_signal   [1.0 if found, else 0.0]

[Multipliers applied after sum:]
  consulting_only → ×0.4
  research_only   → ×0.5
  wrong_domain    → ×0.3
```

| Signal | Inner Weight |
|--------|-------------|
| product_ratio | **0.35** |
| deploy_signal | **0.30** |
| shipper_ratio | **0.20** |
| ownership_signal | **0.15** |

---

### 4F. Cross-Encoder Merge

```
phase4_score = 0.80 × core_score + 0.20 × ce_score
```

| Parameter | Value | Tuning Note |
|-----------|-------|-------------|
| Handcrafted weight | **0.80** | Keeps behavioral/career signals dominant |
| Cross-encoder weight | **0.20** | Validate — reduce to 0.10 if NDCG gain < 2–3 pts |
| Candidates scored by CE | Top **500** by preliminary core_score | Offline only |
| Model | `BAAI/bge-reranker-v2-m3` (130 MB) | — |

---

## Stage 5 — Behavioral Adjustments (Runtime Phase 5)

### Final Score Formula

```
combined_mult  = avail_mult × penalty_mult × logistical_mult
combined_mult  = max(combined_mult, 0.25)          ← floor: score cannot drop >75%

logistical_mult = notice_mult × loc_mult × seniority_mult × writing_mult
logistical_mult = max(logistical_mult, 0.75)        ← floor: logistics alone cannot drop >25%

final_score = phase4_score × combined_mult
            + ninety_day_bonus                      ← additive (+0.00 to +0.08)
            + social_boost                          ← additive (+0.00 to +0.12)
```

---

### 5A. Availability Multiplier (×, STRONG GATE)

Uses: `last_active_date` → `days_inactive`, `recruiter_response_rate`, `open_to_work_flag`

| Condition | Multiplier |
|-----------|-----------|
| days_inactive ≤ **30** AND response_rate ≥ **0.70** AND open_to_work == True | **×1.15** |
| days_inactive ≤ **90** AND response_rate ≥ **0.50** | **×1.05** |
| days_inactive > **180** OR response_rate < **0.15** | **×0.70** |
| all other cases | **×0.90** |

---

### 5B. Soft Penalties (×, STRONG GATE — compounded into penalty_mult)

| Flag / Signal | Condition | Multiplier |
|--------------|-----------|-----------|
| `title_velocity_flag` | avg tenure < 18 months AND ≥ 3 roles | **×0.80** |
| `code_stopped` | VP/Architect/Director title AND yoe > 8 | **×0.75** |
| `langchain_only_flag` | LangChain wrapper + no pre-LLM ML + AI months < 12 | **×0.45** ⚠ |
| `preferred_work_mode == "remote"` | Stated remote preference (role is hybrid) | **×0.85** |
| `research_only` | No engineering titles in career | **×0.40** ⚠ |
| `wrong_domain` | CV/speech/robotics, no NLP/IR | **×0.50** |
| `consistency_score` | See §3D formula | **×[0.30–1.00]** |
| `closed_source_flag` | yoe ≥ 5 AND no external validation | **×0.80** |

> ⚠ Validate against `metadata/validation_set.json` before trusting — may over-penalise edge cases.

---

### 5C. Logistical Multipliers (×, CAPPED GROUP — floor 0.75)

**Notice Period** (uses `redrob_signals.notice_period_days`):

| Notice | Multiplier |
|--------|-----------|
| ≤ 30 days | **×1.00** |
| 31–60 days | **×0.95** |
| 61–90 days | **×0.90** |
| 91–120 days | **×0.85** |
| > 120 days | **×0.75** |

**Location** (uses `profile.location`, `profile.country`, `redrob_signals.willing_to_relocate`):

| Location | Willing to Relocate | Multiplier |
|----------|--------------------|-----------| 
| Pune / Noida / Delhi NCR | any | **×1.00** |
| Hyderabad / Mumbai (welcome cities) | Yes | **×1.00** |
| Hyderabad / Mumbai (welcome cities) | No | **×0.98** |
| Bangalore / Chennai / Kolkata etc. | Yes | **×0.98** |
| Bangalore / Chennai / Kolkata etc. | No | **×0.95** |
| Elsewhere in India | Yes | **×0.95** |
| Elsewhere in India | No | **×0.92** |
| Outside India | Yes | **×0.90** |
| Outside India | No | **×0.85** |

**Seniority** (from `bucket_c.seniority_score`, computed in Phase 3 §3C above):

**Writing Signal** (from `bucket_b.writing_signal`, computed in Phase 3 §3B above):

---

### 5D. 90-Day Alignment Bonus (+, ADDITIVE — range 0.0 to +0.08)

```
m1 = retrieval_search / 3.0
m2 = max(vector_db_hybrid, ltr_reranking) / 3.0
m3 = eval_framework / 3.0

readiness = (m1 + m2 + m3) / 3.0
  if all 3 > 0:  readiness = min(readiness + 0.15, 1.0)   [full coverage bonus]
  if only 1 > 0: readiness = max(readiness - 0.10, 0.0)   [single-milestone penalty]
  if none > 0:   readiness = 0.0

alignment = 0.8 × readiness + 0.2 × product_ratio

ninety_day_bonus = 0.08 × alignment
```

| alignment | ninety_day_bonus |
|-----------|-----------------|
| 0.0 (no milestone evidence) | **+0.000** |
| 0.5 (partial coverage) | **+0.040** |
| 1.0 (all 3 milestones + strong product background) | **+0.080** |

---

### 5E. Social Proof Boost (+, ADDITIVE — capped at +0.12)

Uses 9 of the 23 Redrob signals that are **not already captured by availability or logistical multipliers**.
For the full 23-signal coverage map, see §Complete Redrob Signals Coverage below (17 of 23 total used across all stages).

| Signal | Source Field | Threshold | Boost |
|--------|-------------|-----------|-------|
| GitHub activity | `github_activity_score` | > **60** | **+0.03** |
| Saved by recruiters | `saved_by_recruiters_30d` | > **5** | **+0.04** |
| Profile views | `profile_views_received_30d` | > **20** | **+0.01** |
| Endorsements | `endorsements_received` | > **20** | **+0.01** |
| Interview completion | `interview_completion_rate` | > **0.80** | **+0.02** |
| Offer acceptance | `offer_acceptance_rate` | > **0.70** | **+0.01** |
| Profile completeness | `profile_completeness_score` | > **80** | **+0.01** |
| LinkedIn connected | `linkedin_connected` | == True | **+0.01** |
| Response speed | `avg_response_time_hours` ≤ **4.0** AND `recruiter_response_rate` ≥ **0.60** | combined | **+0.01** |
| **Total cap** | | | **≤ +0.12** |

---

## Complete Redrob Signals Coverage (23 total)

| # | Signal | Status | Where Used |
|---|--------|--------|-----------|
| 1 | `profile_completeness_score` | ✅ Used | Social boost +0.01 |
| 2 | `signup_date` | ❌ Not used | — |
| 3 | `last_active_date` | ✅ Used | Availability mult + ghost filter |
| 4 | `open_to_work_flag` | ✅ Used | Availability mult + ghost filter |
| 5 | `profile_views_received_30d` | ✅ Used | Social boost +0.01 |
| 6 | `applications_submitted_30d` | ✅ Used | Ghost filter |
| 7 | `recruiter_response_rate` | ✅ Used | Availability mult + ghost filter + response-speed boost |
| 8 | `avg_response_time_hours` | ✅ Used | Social boost +0.01 (speed signal) |
| 9 | `skill_assessment_scores` | ✅ Used | Bucket A evidence score +0.5 bonus |
| 10 | `connection_count` | ❌ Not used | Low signal for this role |
| 11 | `endorsements_received` | ✅ Used | Social boost +0.01 |
| 12 | `notice_period_days` | ✅ Used | Notice modifier (logistical ×) |
| 13 | `expected_salary_range_inr_lpa` | ❌ Not used | Not in submission spec |
| 14 | `preferred_work_mode` | ✅ Used | Soft penalty ×0.85 if remote-only |
| 15 | `willing_to_relocate` | ✅ Used | Location modifier (logistical ×) |
| 16 | `github_activity_score` | ✅ Used | Social boost +0.03; external_validation |
| 17 | `search_appearance_30d` | ❌ Not used | Similar to profile_views, skipped |
| 18 | `saved_by_recruiters_30d` | ✅ Used | Social boost +0.04 |
| 19 | `interview_completion_rate` | ✅ Used | Social boost +0.02 |
| 20 | `offer_acceptance_rate` | ✅ Used | Social boost +0.01 |
| 21 | `verified_email` | ❌ Not used | Near-binary noise |
| 22 | `verified_phone` | ❌ Not used | Near-binary noise |
| 23 | `linkedin_connected` | ✅ Used | Social boost +0.01 |

**Total: 17 of 23 signals used**

---

## Score Ranges — End-to-End

| Score / Variable | Min | Max | Notes |
|-----------------|-----|-----|-------|
| Bucket A evidence per domain | 0 | 3 | Integer + 0.5 assessment bonus |
| consistency_score | 0.30 | 1.00 | 0.30 when all expert claims contradicted |
| product_builder_score | 0.00 | 1.00 | After disqualifier multipliers |
| must_have_score | 0.00 | 1.00 | Capped at 0.5/0.60 if no retrieval evidence |
| core_score | 0.00 | 1.00 | 4-component weighted sum |
| phase4_score | 0.00 | ~1.00 | 80% core + 20% cross-encoder |
| combined_mult | **0.25** | ~1.15 | Hard floor 0.25 |
| logistical_mult | **0.75** | 1.00 | Hard floor 0.75 |
| ninety_day_bonus | 0.00 | **+0.08** | Additive |
| social_boost | 0.00 | **+0.12** | Additive, capped |
| **final_score** | **~0.25×** | **~1.20** | phase4 × combined_mult + bonuses |

---

## Output Format

| Column | Type | Constraint |
|--------|------|-----------|
| `candidate_id` | string | Must match `^CAND_[0-9]{7}$` |
| `rank` | int 1–100 | Each value exactly once |
| `score` | float | Non-increasing with rank; not all identical |
| `reasoning` | string (1–2 sentences) | No hallucinations; tone matches rank |

**Tie-breaking:** sort by `(-final_score, candidate_id)` — ascending candidate_id is deterministic.

---

## Tuning Cheat Sheet

| To achieve... | Change this |
|--------------|------------|
| Rank active candidates higher | ↑ availability_mult best-case (currently ×1.15) |
| Reduce penalty for inactive but strong candidates | ↑ combined_mult floor (currently 0.25) |
| Make retrieval skills more dominant | ↑ must-have outer weight (currently 0.55) |
| Make product background matter more | ↑ product_builder outer weight (currently 0.20) |
| Soften LangChain-only penalty | ↑ langchain_only multiplier (currently ×0.45 → try ×0.60) |
| Wider seniority band accepted | Adjust seniority_score table thresholds |
| Reward milestone-ready candidates more | ↑ ninety_day_bonus cap (currently 0.08 → try 0.12) |
| Reduce Bangalore vs Pune gap | ↑ INDIA_ADJACENT multipliers |
| Let social signals matter more | ↑ social_boost cap (currently 0.12) |
| Reduce CE influence | ↓ cross_encoder_weight (currently 0.20 → try 0.10) |

---

## Configuration File: `weights.yaml`

> **Option A — Dynamic Config Injection (confirmed design).**
> The Python source files (`scorer.py`, `behavioral.py`, `features.py`) keep readable default
> literal values in comments, but at runtime they load all weights, thresholds, and multipliers
> from `weights.yaml` into a single `cfg` dictionary.
> **To tune:** edit `weights.yaml` only → re-run `rank.py`. No Python code changes needed.

```yaml
# weights.yaml — Evidence-Rank tuning parameters
# Option A: loaded once at startup in rank.py and preprocess.py
# Usage: cfg = yaml.safe_load(open("weights.yaml"))
# All values below correspond exactly to the numbers described in this variables.md

# ── Phase 4: Core Score outer weights ───────────────────────────────────────
phase4:
  must_have_weight:      0.55
  nice_to_have_weight:   0.10
  career_quality_weight: 0.15
  product_builder_weight: 0.20
  cross_encoder_weight:  0.20   # Validate: reduce to 0.10 if NDCG gain < 2-3 pts

# ── Phase 4: Must-Have inner weights (sum = 0.60) ───────────────────────────
must_have:
  retrieval_search_w: 0.25
  vector_db_hybrid_w: 0.20
  eval_framework_w:   0.15
  no_retrieval_cap:   0.50     # Cap must_have_raw if no retrieval evidence

# ── Phase 4: Nice-to-Have inner weights (sum = 0.10) ────────────────────────
nice_to_have:
  ltr_reranking_w:   0.07
  llm_integration_w: 0.03

# ── Phase 4: Career Quality inner weights (sum = 0.15) ──────────────────────
career_quality:
  sys_experience_w:    0.08
  experience_recency_w: 0.04
  depth_signal_w:      0.03
  consulting_mult:     0.40
  research_mult:       0.50
  wrong_domain_mult:   0.30

# ── Phase 4: Product Builder inner weights (sum = 1.00 before disqualifiers) ─
product_builder:
  product_ratio_w:    0.35
  deploy_signal_w:    0.30
  shipper_ratio_w:    0.20
  ownership_signal_w: 0.15
  consulting_mult:    0.40
  research_mult:      0.50
  wrong_domain_mult:  0.30

# ── Phase 5: Availability multiplier thresholds ─────────────────────────────
availability:
  best_days_threshold:      30
  best_response_threshold:  0.70
  good_days_threshold:      90
  good_response_threshold:  0.50
  bad_days_threshold:       180
  bad_response_threshold:   0.15
  best_mult:  1.15
  good_mult:  1.05
  bad_mult:   0.70
  default_mult: 0.90

# ── Phase 5: Notice period thresholds ───────────────────────────────────────
notice:
  tier1_days: 30
  tier2_days: 60
  tier3_days: 90
  tier4_days: 120
  mult_t1: 1.00
  mult_t2: 0.95
  mult_t3: 0.90
  mult_t4: 0.85
  mult_t5: 0.75

# ── Phase 5: Soft penalties ──────────────────────────────────────────────────
penalties:
  title_velocity_mult:  0.80
  code_stopped_mult:    0.75
  langchain_only_mult:  0.45
  remote_only_mult:     0.85
  research_only_mult:   0.40
  wrong_domain_mult:    0.50
  closed_source_mult:   0.80
  consistency_scale:    0.70   # consistency = 1.0 - consistency_scale × (contradictions/checks)

# ── Phase 5: Logistical group floors ─────────────────────────────────────────
floors:
  logistical_floor: 0.75   # floor on notice × location × seniority × writing
  combined_floor:   0.25   # floor on avail × penalties × logistical

# ── Phase 5: 90-day alignment bonus ──────────────────────────────────────────
ninety_day:
  bonus_scale:           0.08   # ninety_day_bonus = bonus_scale × alignment
  full_coverage_bonus:   0.15   # added to readiness when all 3 milestones covered
  single_milestone_penalty: 0.10 # subtracted when only 1 milestone covered
  readiness_weight:      0.80
  product_ratio_weight:  0.20

# ── Phase 5: Social proof boost thresholds ───────────────────────────────────
social:
  github_threshold:       60
  github_boost:           0.03
  recruiter_saves_threshold: 5
  recruiter_saves_boost:  0.04
  profile_views_threshold: 20
  profile_views_boost:    0.01
  endorsements_threshold: 20
  endorsements_boost:     0.01
  interview_rate_threshold: 0.80
  interview_rate_boost:   0.02
  offer_acceptance_threshold: 0.70
  offer_acceptance_boost: 0.01
  profile_completeness_threshold: 80
  profile_completeness_boost: 0.01
  linkedin_boost:         0.01
  response_speed_hours:   4.0
  response_speed_rate:    0.60
  response_speed_boost:   0.01
  boost_cap:              0.12   # Max total social boost

# ── Phase 1: Ghost filter thresholds ─────────────────────────────────────────
ghost:
  days_inactive_threshold: 365
  response_rate_threshold: 0.05

# ── Phase 1: Honeypot thresholds ─────────────────────────────────────────────
honeypot:
  skill_duration_grace_months: 6    # skill_duration > yoe_months + grace → honeypot
  role_duration_grace_months: 12    # role_duration > yoe_months + grace → honeypot
  assessment_fail_threshold: 40     # expert skill with score < threshold → honeypot

# ── Phase 1: Disqualifier thresholds ─────────────────────────────────────────
disqualifiers:
  consulting_only_threshold: 0.0  # product_ratio must be exactly 0.0
  langchain_ai_months_max: 12     # AI skill months < max + no pre-LLM → flag
  closed_source_yoe_min: 5        # years of experience before checking external validation
  code_stopped_yoe_min: 8         # yoe threshold for code_stopped flag
  title_velocity_months: 18       # avg tenure < this AND >= 3 roles → flag
  title_velocity_min_roles: 3

# ── Phase 2: Retrieval ───────────────────────────────────────────────────────
retrieval:
  faiss_top_k:     2000
  bm25_top_k:      2000
  rrf_k:           60
  pool_size:       3000
  activity_boost_rank: 200   # Soft rank boost for active seekers pre-RRF
  cross_encoder_top_n: 500   # Candidates scored by CE offline

# ── Phase 3: Evidence scoring ─────────────────────────────────────────────────
evidence:
  assessment_boost:       0.50   # Added to bucket A score if assessment >= threshold
  assessment_threshold:   70     # Minimum platform score to earn boost
  production_density_cap: 5      # deploy_signal = count / cap, capped at 1.0
  depth_signal_cap:       2      # roles_with_retrieval / cap, capped at 1.0
  writing_long_threshold: 150    # >= chars → writing_signal = 1.00
  writing_short_threshold: 60    # >= chars → writing_signal = 0.95, else 0.90
```
