# Evidence-Rank: Complete System Deep Dive
> **Project:** Redrob Hackathon — Intelligent Candidate Discovery & Ranking Engine  
> **Version:** 3.3.0 | Source: [plan.md](file:///d:/GitHub/evidence-rank/plan.md) + [variables.md](file:///d:/GitHub/evidence-rank/variables.md)

---

# 1. One Minute Explanation

* **The Problem:** A hiring platform (Redrob) has 100,000 candidate profiles. They need a computer to find and rank the top 100 best fits for one specific job — a Senior AI Engineer on a founding team — the same way a skilled recruiter would, not just by keyword matching.
* **Who Uses It:** Competition judges (hackathon), and conceptually: any recruiter or hiring platform that needs to rank candidates at scale.
* **What Goes In:** A `candidates.jsonl` file with 100,000 candidate profiles (title, career history, skills, behavioral signals like last login date) + one Job Description (JD) for a Senior AI Engineer.
* **What Comes Out:** A `submission.csv` file with exactly 100 rows — the top 100 candidates, ranked best-fit first, each with a 1–2 sentence explanation of why they ranked there.
* **Why It Exists:** Naive keyword search fails — a candidate who built a recommendation engine at Flipkart is a great fit even if they never say "RAG" or "Pinecone." Equally, a candidate with every AI keyword whose entire career is at TCS is not a fit. This system reasons about the *gap between what the JD says and what the JD means.*
* **The Hard Constraint:** The ranking step must finish in **under 5 minutes** on a standard CPU with no internet. This forces all heavy ML work (embeddings, cross-encoder inference) to run offline beforehand.
* **The Key Insight:** Evidence beats claims. A skill listed in the "Skills" section scores 1/3. That same skill appearing in a career description *with production/scale signals* scores 3/3. The system rewards proof over assertion.
* **What Makes It Different:** It uses *two* JD vectors (technical vocabulary + ideal-candidate narrative), a 3-level evidence scoring system (skills mention → career project → production deployment), a Consistency Score to detect keyword stuffers, and soft multipliers with floors (not hard cutoffs) for behavioral and logistical signals — mirroring how a real recruiter thinks.
* **The Hackathon Scoring:** NDCG@10 = 50% of the score. Your top 10 picks matter more than picks 11–100 combined.

---

# 2. Elevator Pitch

> "We built a candidate ranking engine that thinks like a senior recruiter, not a search engine. It doesn't just match keywords — it reads career history to find evidence of production deployments, detects keyword stuffers whose claims don't match their career, flags unreachable passive candidates, and explains each ranking in plain English tied to the specific milestones in the job description. The whole ranking step runs in under 5 minutes on a laptop, because all the heavy AI work happens offline beforehand."

---

# 3. End-to-End Pipeline

## Phase 0 — JD Intelligence *(Offline, run once)*

| | |
|---|---|
| **Purpose** | Decode the job description into machine-readable rules, embeddings, and keywords |
| **Input** | Raw JD text for "Senior AI Engineer, Founding Team, Redrob AI" |
| **Processing** | Parse JD into 5 typed buckets (must-haves, nice-to-haves, hard disqualifiers, soft negatives, logistics); embed it into **two** semantic vectors using BGE-M3; build BM25 keyword list |
| **Output** | `jd_config.json`, `jd_skills_vector.npy`, `jd_ideal_vector.npy`, `jd_keywords.json` |

**Why two vectors?** The skills-focused vector catches technically-worded profiles ("FAISS, NDCG, hybrid search"). The ideal-candidate vector catches plain-language profiles ("built a recommendation system at a product company"). Each type of candidate is invisible to the other vector alone.

---

## Phase 1 — Corpus Preprocessing + Honeypot Detection *(Offline, run once)*

| | |
|---|---|
| **Purpose** | Embed all 100K candidates, build search indexes, flag fraudulent/unreachable profiles |
| **Input** | `candidates.jsonl` (100,000 profiles) |
| **Processing** | Flatten each profile into text → embed with BGE-M3 → store in FAISS index; build BM25 index; run 4 honeypot checks; run ghost-profile filter (4 conditions, AND logic); run disqualifier tagging (consulting-only, research-only, wrong-domain) |
| **Output** | `faiss_index.bin`, `bm25_index.pkl`, `candidate_ids.json`, `candidate_texts.pkl`, `candidate_flags.parquet` |

Also runs sub-phases offline:
- **Phase 1d:** RRF retrieval → `retrieval_scores.parquet`
- **Phase 1c:** Feature extraction on top 5,000 → `candidate_features.parquet`
- **Phase 1e:** Cross-encoder on top 5,000 RRF candidates → `cross_encoder_scores.parquet`

---

## Phase 2 — Multi-Signal Retrieval *(Runtime, < 5 seconds)*

| | |
|---|---|
| **Purpose** | Narrow 100K candidates down to 3,000–5,000 worth scoring in detail |
| **Input** | `retrieval_scores.parquet` (precomputed RRF scores) |
| **Processing** | Load parquet → filter to top 5,000 by RRF score (no FAISS or BM25 calls at runtime) |
| **Output** | List of 3,000–5,000 candidate IDs with RRF scores |

*The actual FAISS + BM25 + RRF computation happened offline (Phase 1d). Runtime Phase 2 is a pure pandas read.*

---

## Phase 3 — Feature Extraction *(Runtime, < 5 seconds)*

| | |
|---|---|
| **Purpose** | Extract structured evidence features for every retrieved candidate |
| **Input** | `candidate_features.parquet` (precomputed offline) |
| **Processing** | Load parquet → join features onto candidate IDs (no regex computation at runtime) |
| **Output** | Three evidence buckets per candidate: **A** (technical skill evidence, 0–3 per domain), **B** (career quality signals), **C** (JD fit gaps), plus a **Consistency Score** |

---

## Phase 4 — Core Scoring + Cross-Encoder Rerank *(Runtime, < 30 seconds)*

| | |
|---|---|
| **Purpose** | Compute a technical fit score and refine the top 500 with semantic cross-encoder judgment |
| **Input** | Bucket A/B/C features, `cross_encoder_scores.parquet` |
| **Processing** | Apply 4-component weighted formula (Must-Have 55% + Nice-to-Have 10% + Career Quality 15% + Product Builder 20%) → merge precomputed cross-encoder scores (80% handcrafted + 20% CE) |
| **Output** | `final_phase4_score` per candidate; narrowed to top 200–300 |

---

## Phase 5 — Behavioral Re-ranking + Penalization *(Runtime, < 10 seconds)*

| | |
|---|---|
| **Purpose** | Convert technical fit score into a practical hiring score |
| **Input** | Phase 4 scores, behavioral signals from `candidate_features.parquet` |
| **Processing** | Apply multiplicative chain (availability × penalties × logistical group — with two hard floors) then add bonuses (90-day alignment bonus + social proof boost) |
| **Output** | `final_score` per candidate; narrowed to top 100 |

---

## Phase 6 — Reason Generation *(Runtime, < 30 seconds)*

| | |
|---|---|
| **Purpose** | Write a 1–2 sentence honest explanation for why each candidate ranked where they did |
| **Input** | Final scores, ranks, career evidence snippets (stored in `candidate_features.parquet`) |
| **Processing** | Template-based string assembly from verbatim career snippets; tone calibrated to rank tier (positive at 1–30, honest gaps mandatory at 71–100) |
| **Output** | `reasoning` column in `submission.csv`; no LLM called |

---

## Phase 7 — Manual Validation *(Pre-submission, outside 5-min limit)*

| | |
|---|---|
| **Purpose** | Human sanity check before submitting |
| **Input** | `submission.csv` |
| **Processing** | Top-20 manual read; obvious-fit audit; honeypot audit; reasoning quality check; format validation |
| **Output** | Final `BuriBuri.csv` upload |

---

# 4. Candidate Elimination / Filtering Logic

| Stage | Why Candidate Is Removed | Example |
|---|---|---|
| **Phase 1 — Honeypot Check 1** | Skill claimed for more months than total career length (+6mo grace) | Claims "Python: 120 months" but total YoE = 5 years (60 months) |
| **Phase 1 — Honeypot Check 2** | Expert/Advanced skill with zero months of usage | Claims "FAISS: Expert" with `duration_months = 0` |
| **Phase 1 — Honeypot Check 3** | Expert/Advanced skill with platform assessment score < 40 | Claims "NLP: Expert" but scored 22/100 on Redrob's NLP test |
| **Phase 1 — Honeypot Check 4** | A single job role lasting longer than total career YoE (+12mo grace) | Role at Company X = 84 months, but total YoE = 4 years (48 months) |
| **Phase 1 — Ghost Pre-filter** | Inactive > 365 days AND response rate < 5% AND not open to work AND zero applications in 30 days (all four conditions must hold) | Last login 2 years ago, 2% recruiter response rate, not actively seeking |
| **Phase 2 — RRF Threshold** | Candidate not retrieved by any of the 3 retrievers (FAISS ×2 + BM25) | Pure Excel/BI analyst with no ML vocabulary anywhere in profile |
| **Phase 4 — Score Threshold** | Core score too low to make top 200–300 cutoff | Candidate retrieved but has no retrieval, vector DB, or system evidence |
| **Phase 5 — Final Cutoff** | Final score after all multipliers too low to make top 100 | Technically decent but inactive 300 days, 90-day notice, outside India, remote-only |

> **Key nuance:** Honeypots and ghosts are **hard-zeroed** (score = 0.0). Everything else is **soft** — strong technical evidence can overcome behavioral penalties. Two hard floors prevent complete collapse: combined multiplier floor = 0.25; logistical group floor = 0.75. Only ~1–3% are ghost-filtered; ~0.08% are honeypots.

---

# 5. Ranking / Scoring Logic

## Layer 1: Retrieval (Phase 2) — who enters the pipeline

**Reciprocal Rank Fusion (RRF):**

$$\text{RRF}(d) = \sum_i \frac{1}{60 + \text{rank}_i(d)}$$

Three ranked lists are fused: FAISS (skills vector), FAISS (ideal-candidate vector), BM25. A candidate appearing in all three lists gets a much higher score than one appearing in just one.

---

## Layer 2: Core Score (Phase 4) — technical fit

$$\text{Core Score} = 0.55 \times S_{\text{must-have}} + 0.10 \times S_{\text{nice-to-have}} + 0.15 \times S_{\text{career quality}} + 0.20 \times S_{\text{product builder}}$$

### Must-Have Score (55% of core_score):

$$S_{\text{must-have}} = \frac{0.25 \times S_{\text{retrieval}} + 0.20 \times S_{\text{vectordb}} + 0.15 \times S_{\text{eval}}}{0.60}$$

Each domain score is 0–1 (normalized from the 0–3 Bucket A evidence score).

**Hard cap:** If `retrieval_search`, `vector_db_hybrid`, and `sys_experience_score` are all 0, must-have raw score is capped at 0.50.

### Nice-to-Have Score (10% of core_score):

$$S_{\text{nice-to-have}} = \frac{0.07 \times S_{\text{ltr}} + 0.03 \times S_{\text{llm}}}{0.10}$$

### Career Quality Score (15% of core_score):

$$S_{\text{career}} = \frac{0.08 \times \text{sys\_experience\_score} + 0.04 \times \text{experience\_recency} + 0.03 \times \text{depth\_signal}}{0.15}$$

**Career quality multipliers (applied before normalization):**
- Consulting-only career: ×0.4
- Research-only career: ×0.5
- Wrong domain (CV/speech, no NLP): ×0.3

### Product Builder Score (20% of core_score):

$$S_{\text{product builder}} = 0.35 \times \text{product\_ratio} + 0.30 \times \text{deploy\_signal} + 0.20 \times \text{shipper\_ratio} + 0.15 \times \text{ownership\_signal}$$

**Product builder multipliers (applied after sum):**
- Consulting-only career: ×0.4
- Research-only career: ×0.5
- Wrong domain: ×0.3

### Cross-Encoder Merge:

$$\text{Phase4 Score} = 0.80 \times \text{Core Score} + 0.20 \times \text{Cross-Encoder Score}$$

Cross-encoder (`bge-reranker-v2-m3`) sees JD + candidate simultaneously — catches semantic nuances cosine similarity misses. Kept at 20% so handcrafted behavioral signals remain dominant.

---

## Layer 3: Final Score (Phase 5) — practical hiring score

```
logistical_mult  = notice_mult × loc_mult × seniority_mult × writing_mult
logistical_mult  = max(logistical_mult, 0.75)      ← floor: logistics group cannot drop score > 25%

combined_mult    = avail_mult × penalty_mult × logistical_mult
combined_mult    = max(combined_mult, 0.25)         ← floor: total multiplicative chain cannot drop score > 75%

final_score      = phase4_score × combined_mult
                 + ninety_day_bonus                 ← additive, +0.00 to +0.08
                 + social_boost                     ← additive, +0.00 to +0.12 (capped)
```

> **Why floors?** The two hard floors prevent any single signal cluster (availability + penalties + logistics) from completely collapsing a candidate with strong technical evidence. Strong evidence must always be able to show through.

> **Why additive for 90-day and social?** These are *rewards*, not gates. Missing milestone evidence ≠ wrong hire. Making them additive means they lift good candidates without blocking anyone.

### Multiplier Ranges:

| Multiplier | Type | Range | Best Case | Worst Case |
|---|---|---|---|---|
| Availability | × STRONG GATE | 0.70–1.15 | Active ≤30d, response ≥70%, open to work | Active >180d or response <15% |
| Soft Penalties (combined) | × STRONG GATE | ~0.30–1.00 | No flags triggered | LangChain-only (×0.45) × research-only (×0.40) × other flags |
| Notice Period | × LOGISTICAL | 0.75–1.00 | ≤30 days | >120 days |
| Location | × LOGISTICAL | 0.85–1.00 | Pune/Noida/Delhi NCR | Outside India, won't relocate |
| Seniority | × LOGISTICAL | 0.40–1.00 | 5–9 years YoE | <3 years YoE |
| Writing Signal | × LOGISTICAL | 0.90–1.00 | Avg description ≥150 chars | Avg description <60 chars |
| Logistical group floor | — | min 0.75 | — | notice × location × seniority × writing cannot reduce score by >25% |
| Combined mult floor | — | min 0.25 | — | Full chain cannot reduce score by >75% |
| 90-Day Alignment | + ADDITIVE | +0.00 to +0.08 | All 3 milestones + strong product background | No milestone evidence |
| Social Proof Boost | + ADDITIVE | +0.00 to +0.12 | Max signals active | No signals |

### Soft Penalties (detail):

| Flag / Signal | Condition | Multiplier |
|---|---|---|
| `title_velocity_flag` | avg tenure < 18 months AND ≥ 3 roles | **×0.80** |
| `code_stopped` | VP/Architect/Director title AND yoe > 8 | **×0.75** |
| `langchain_only_flag` | LangChain wrapper + no pre-LLM ML + AI months < 12 | **×0.45** ⚠ |
| `preferred_work_mode == "remote"` | Stated remote preference (role is hybrid) | **×0.85** |
| `research_only` | No engineering titles in career | **×0.40** ⚠ |
| `wrong_domain` | CV/speech/robotics, no NLP/IR | **×0.50** |
| `consistency_score` | `1.0 - 0.7 × (contradicted/total)` | **×0.30–1.00** |
| `closed_source_flag` | yoe ≥ 5 AND no external validation | **×0.80** |

> ⚠ Validate ×0.45 and ×0.40 against `metadata/validation_set.json` before submitting — may over-penalise edge cases.

### Social Proof Boost (additive, capped at +0.12):

| Signal | Source Field | Threshold | Boost |
|---|---|---|---|
| Saved by recruiters | `saved_by_recruiters_30d` | > 5 | **+0.04** |
| GitHub activity | `github_activity_score` | > 60 | **+0.03** |
| Interview completion | `interview_completion_rate` | > 0.80 | **+0.02** |
| Profile views | `profile_views_received_30d` | > 20 | **+0.01** |
| Endorsements | `endorsements_received` | > 20 | **+0.01** |
| Offer acceptance | `offer_acceptance_rate` | > 0.70 | **+0.01** |
| Profile completeness | `profile_completeness_score` | > 80 | **+0.01** |
| LinkedIn connected | `linkedin_connected` | == True | **+0.01** |
| Response speed | `avg_response_time_hours` ≤ 4.0 AND `recruiter_response_rate` ≥ 0.60 | combined | **+0.01** |
| **Total cap** | | | **≤ +0.12** |

---

## Bucket A Evidence Scoring (0–3 per domain)

| Score | Meaning |
|---|---|
| 0 | Skill not present anywhere |
| 1 | Mentioned in skills section only (claim without evidence) |
| 2 | Appears in career description (project-level proof) |
| 3 | Career description + production/scale signals ("shipped to real users", "p99 latency", "billion queries") |

**Bonus:** Platform-verified assessment score ≥70 on a target skill → +0.5 (capped at 3)

---

# 6. Data Model

## Candidate

The atomic unit. Everything flows from this.

| Field Path | Type | Why It Matters |
|---|---|---|
| `candidate_id` | string | Primary key (`CAND_XXXXXXX`) — ties everything together |
| `profile.years_of_experience` | float | Used in seniority scoring and honeypot detection |
| `profile.current_title` | string | Checked for "code-stopped" (VP/Architect titles) |
| `profile.location` / `profile.country` | string | Location modifier calculation |
| `career_history[].description` | string | **Primary evidence source.** Where all 0–3 evidence scores come from |
| `career_history[].company` | string | Consulting firm detection |
| `career_history[].duration_months` | int | Honeypot check 4; title velocity flag |
| `skills[].name` | string | Baseline skill mention check (score=1) |
| `skills[].proficiency` | string | `expert`/`advanced` triggers consistency score checks |
| `skills[].duration_months` | int | Honeypot check 1; consistency score check 2 |
| `redrob_signals.skill_assessment_scores` | dict | Platform-verified scores; boost evidence score by +0.5 |
| `redrob_signals.last_active_date` | date | Ghost filter; availability multiplier |
| `redrob_signals.recruiter_response_rate` | float | Ghost filter; availability multiplier; response-speed boost |
| `redrob_signals.open_to_work_flag` | bool | Ghost filter; availability multiplier |
| `redrob_signals.notice_period_days` | int | Notice period modifier |
| `redrob_signals.github_activity_score` | float | Social proof boost (+0.03); external validation signal |
| `redrob_signals.saved_by_recruiters_30d` | int | Social proof boost (+0.04) |
| `redrob_signals.interview_completion_rate` | float | Social proof boost (+0.02) |
| `redrob_signals.offer_acceptance_rate` | float | Social proof boost (+0.01) |
| `redrob_signals.profile_completeness_score` | float | Social proof boost (+0.01) |
| `redrob_signals.profile_views_received_30d` | int | Social proof boost (+0.01) |
| `redrob_signals.avg_response_time_hours` | float | Response-speed social boost (+0.01 combined) |
| `redrob_signals.endorsements_received` | int | Social proof boost (+0.01) |
| `redrob_signals.linkedin_connected` | bool | Social proof boost (+0.01) |
| `redrob_signals.willing_to_relocate` | bool | Location modifier |
| `redrob_signals.preferred_work_mode` | string | Soft penalty ×0.85 if remote-only |
| `redrob_signals.applications_submitted_30d` | int | Ghost filter condition |

## Artifact Files (Precomputed)

| File | What It Stores | Why It Matters |
|---|---|---|
| `candidate_flags.parquet` | `is_honeypot`, `is_ghost`, `product_ratio`, `consulting_only`, `research_only`, `wrong_domain` per candidate | Hard exclusions + career quality inputs |
| `retrieval_scores.parquet` | `candidate_id` + `rrf_score` for top 5,000 | The gate into the full pipeline |
| `candidate_features.parquet` | All Bucket A/B/C values + snippets + behavioral + consistency score | Everything Phase 4/5/6 needs |
| `cross_encoder_scores.parquet` | `candidate_id` + `ce_raw_score` + normalized `ce_score` for top 5,000 | Semantic tiebreaker at Phase 4 |

---

# 7. Architecture Diagram (Text Form)

```
JD Text (one-time input)
     │
     ▼
[Phase 0: JD Intelligence] ──────────────────────────────────────────── OFFLINE
     │  BGE-M3 embedder
     │  → jd_skills_vector.npy
     │  → jd_ideal_vector.npy
     │  → jd_keywords.json
     │  → jd_config.json
     │
candidates.jsonl (100K profiles)
     │
     ▼
[Phase 1: Corpus Preprocessing] ─────────────────────────────────────── OFFLINE
     │  BGE-M3 × 100K profiles → FAISS IndexFlatIP
     │  BM25Okapi index
     │  Honeypot 4-check filter
     │  Ghost 4-condition filter (AND logic)
     │  Disqualifier tagging (consulting/research/domain)
     │  → faiss_index.bin
     │  → bm25_index.pkl
     │  → candidate_flags.parquet
     │
     ▼
[Phase 1d: RRF Retrieval] ────────────────────────────────────────────── OFFLINE
     │  FAISS (skills vector) → top 2,000
     │  FAISS (ideal vector)  → top 2,000
     │  BM25 (keyword list)   → top 2,000
     │  RRF fusion (k=60)     → top 3,000–5,000 pool
     │  → retrieval_scores.parquet
     │
     ▼
[Phase 1c: Feature Extraction] ──────────────────────────────────────── OFFLINE
     │  Regex patterns on career descriptions
     │  Bucket A: skill evidence 0–3 per domain
     │  Bucket B: career quality signals + product_builder_score
     │  Bucket C: JD fit gaps + seniority
     │  Consistency Score
     │  Preliminary core score (for CE candidate selection)
     │  → candidate_features.parquet
     │
     ▼
[Phase 1b: Cross-Encoder Inference] ─────────────────────────────────── OFFLINE
     │  bge-reranker-v2-m3 on top 5,000 RRF candidates
     │  ~2.5 min on CPU
     │  → cross_encoder_scores.parquet
     │
════════════════════════════════════════════════════════════════════════
                          RUNTIME STARTS HERE (≤5 min budget)
════════════════════════════════════════════════════════════════════════
     │
     ▼
[Phase 2: Load Retrieval] ─────────────────────────────────────────── RUNTIME
     │  pandas.read_parquet(retrieval_scores.parquet)
     │  filter to top 5,000 by rrf_score
     │  ~5 seconds
     │
     ▼
[Phase 3: Load Features] ──────────────────────────────────────────── RUNTIME
     │  pandas.read_parquet(candidate_features.parquet)
     │  join on candidate_id
     │  ~5 seconds
     │
     ▼
[Phase 4: Core Scoring + CE Merge] ───────────────────────────────── RUNTIME
     │  4-component formula: 55% must-have + 10% nice-to-have
     │                      + 15% career quality + 20% product builder
     │  Left-join cross_encoder_scores.parquet
     │  Final Phase4 = 0.80 × core + 0.20 × CE
     │  Narrow to top 200–300
     │  ~30 seconds
     │
     ▼
[Phase 5: Behavioral Re-ranking] ─────────────────────────────────── RUNTIME
     │  Logistical mult = notice × location × seniority × writing
     │  logistical_mult = max(logistical_mult, 0.75)   ← floor
     │  combined_mult = avail × penalties × logistical_mult
     │  combined_mult = max(combined_mult, 0.25)        ← floor
     │  final = phase4 × combined_mult
     │         + ninety_day_bonus  (additive, +0 to +0.08)
     │         + social_boost      (additive, +0 to +0.12)
     │  Hard-zero honeypots and ghosts
     │  Narrow to top 100
     │  ~10 seconds
     │
     ▼
[Phase 6: Reason Generation] ─────────────────────────────────────── RUNTIME
     │  Template assembly from career snippets
     │  90-day milestone framing
     │  Rank-appropriate tone
     │  ~30 seconds
     │
     ▼
[submission.csv] — 100 rows: candidate_id, rank, score, reasoning
     │
     ▼
[Phase 7: Manual Validation] ─────────────────────── PRE-SUBMISSION
     │  Top-20 read, obvious-fit audit, honeypot audit, reasoning check
     └─→ BuriBuri.csv  (final upload)
```

**Parallel sandbox path (`app.py`):**
```
Small JSON payload (≤100 candidates)
     ↓
[BM25-only heuristic scoring] (no FAISS, no cross-encoder)
     ↓
Ranked results in <10 seconds (HuggingFace Spaces demo)
```

---

# 8. Business Rules

**Hard rules (score = 0.0, absolute exclusion):**
1. Honeypot: Any one of the 4 structural impossibility checks triggers score = 0.0
2. Ghost: All 4 ghost conditions true simultaneously → score = 0.0

**Must-have gate:**
3. Zero retrieval evidence AND zero vector DB evidence AND zero system semantics evidence → must-have score capped at 0.5 (before normalization)

**Soft disqualifiers (multipliers, not hard rejections):**
4. Consulting-only career → career quality ×0.4; product builder ×0.4
5. Research-only background → career quality ×0.5; product builder ×0.5; soft penalty ×0.40
6. Wrong domain (CV/speech without NLP) → career quality ×0.3; product builder ×0.3; soft penalty ×0.50
7. LangChain-only AI (<12 months, no pre-LLM background) → soft penalty ×0.45
8. Code-stopped (VP/Architect + YoE > 8) → soft penalty ×0.75
9. Title-chaser (avg tenure <18mo across 3+ jobs) → soft penalty ×0.80
10. Remote-only preference (role is hybrid) → soft penalty ×0.85
11. Closed-source only (5+ years, no GitHub/papers/talks) → soft penalty ×0.80
12. Consistency Score penalty (expert claims contradicted by career) → multiplier ×0.30–1.00

**Floor rules (prevent extreme score collapse):**
13. Logistical group (notice × location × seniority × writing) floor = 0.75
14. Combined multiplier chain (availability × penalties × logistical) floor = 0.25

**Availability rules:**
15. Active ≤30d + response ≥70% + open to work → ×1.15
16. Active >180d or response <15% → ×0.70 (softened from 0.50 — competition scores fit, not hireability)

**Seniority rules:**
17. 5–9 years YoE = sweet spot (×1.00)
18. <3 years → ×0.40 (nearly disqualifying)
19. YoE is NOT a hard cutoff — strong signals can overcome it

**Evidence scoring rules:**
20. Skill listed in skills section only → score 1 (claim, not evidence)
21. Skill in career description → score 2 (project proof)
22. Skill in career description + production signals → score 3 (deployment proof)
23. Platform assessment score ≥70 on target skill + career evidence → +0.5 bonus

**Tie-breaking:**
24. Equal final scores → sort by `candidate_id` ascending (deterministic)

**90-day plan:**
25. Covering all 3 milestones → readiness +0.15 bonus (capped at 1.0)
26. Covering only 1 milestone → readiness −0.10 penalty (floored at 0.0)
27. Covering 0 milestones → readiness = 0.0
28. Final bonus: `ninety_day_bonus = 0.08 × alignment` (additive, not multiplicative)

---

# 9. Business Rules Explained

**Rule 1–2 (Honeypots & Ghosts):**
Honeypots are fake profiles deliberately injected to test if the system can detect impossible data. A real human cannot have 10 years of Python experience but only 3 years of total career. Ghosts are profiles that are reachable on paper but unreachable in practice — no system should waste a recruiter's time with someone who hasn't logged in for over a year and never responds. Both are zeroed because they corrupt the top-100 quality.

**Rule 3 (Must-Have Cap):**
The JD is unambiguous: the role is for someone who has built retrieval or ranking systems. If a candidate has zero evidence of ever doing this — not even a vague mention — capping their score prevents accidentally promoting a totally irrelevant candidate whose other signals were strong.

**Rule 4–6 (Consulting/Research/Wrong Domain Multipliers):**
The JD explicitly says consulting-only backgrounds are a "bad fit in both directions" and pure researchers who haven't deployed won't be moved forward. Computer vision specialists without NLP need to relearn fundamentals. These are soft (not zero) because someone who spent 80% of their career at TCS but 20% at a product startup is not the same as someone who spent 100% at consulting.

**Rule 7 (LangChain-Only ×0.45):**
This is the harshest soft penalty because the JD is extremely specific: someone who has only used LangChain to call OpenAI APIs for under 12 months with no pre-LLM ML background is not qualified for this role. The system needs strong prior ML foundations, not tutorial-level LLM wrappers.

**Rule 8 (Code-Stopped ×0.75):**
The JD literally says: "This role writes code. Senior engineers who haven't written production code in the last 18 months because they've moved into 'architecture' or 'tech lead' roles — we will probably not move forward." An Architect title with 10+ YoE is a proxy for this risk.

**Rules 13–14 (Floors):**
No single signal cluster should be able to completely override strong technical evidence. A once-great candidate who has been inactive for 7 months should not score near zero if their retrieval/search evidence is exceptional. The 0.25 floor ensures technical merit always shows through.

**Rules 20–22 (Evidence Scoring 0–3):**
The most important design decision in the system. Listing "FAISS" in your skills section proves nothing — you might have just copied it. Writing in your job description that you "deployed a FAISS-based dense retrieval system serving 50K queries/day" is real evidence. The system systematically rewards proof over assertion.

**Rule 24 (Tie-Breaking):**
The competition specification requires unique ranks even for tied scores. Using `candidate_id` ascending is deterministic and fair — no candidate is favored.

**Rules 25–28 (90-Day Plan, Additive):**
The JD describes exactly what the first hire will do in weeks 1, 4, and 9. A candidate who has evidence for all three milestones is inherently more immediately useful. **The bonus is additive (+0 to +0.08)**, not multiplicative — it rewards readiness without penalizing candidates who are excellent fits but whose experience wasn't phrased in milestone-matching language.

---

# 10. Important Algorithms

## Algorithm 1: Reciprocal Rank Fusion (RRF)

**Purpose:** Merge three ranked lists (2×FAISS + 1×BM25) into one without needing to normalize their scores.  
**Inputs:** Three lists of candidate IDs, each sorted by relevance.  
**Output:** A merged dict of `{candidate_id: rrf_score}`.  
**Formula:** For each candidate, sum `1/(60 + rank)` across all lists they appear in.  
**Why k=60:** Dampens the influence of the top few positions and rewards broad multi-list presence over one very-high single-list rank.  
**Plain English:** A candidate who appears at rank 5 in the skills-vector search, rank 10 in the ideal-candidate search, and rank 20 in BM25 scores higher than a candidate who appears at rank 1 in only one list. Multi-dimensional relevance beats single-dimension perfection.  
**Complexity:** O(N) where N = union of all list sizes (~6,000).

---

## Algorithm 2: Honeypot Detection

**Purpose:** Find structurally impossible profiles (fabricated test candidates in the dataset).  
**Inputs:** `skills`, `career_history`, `years_of_experience`, `skill_assessment_scores`.  
**Output:** Boolean — is this candidate a honeypot?  
**4 checks:** (1) Skill duration > total career months + 6; (2) Expert/Advanced skill with 0 months duration; (3) Expert/Advanced skill with assessment score <40; (4) Single role duration > total career months + 12.  
**Plain English:** Checks for things that can't be true — you can't have used Python for 10 years if your entire career is 5 years. If any check fires, score = 0.0, never in top 100.  
**Complexity:** O(S × R) where S = skills count, R = roles count. Microseconds per candidate.

---

## Algorithm 3: Bucket A Evidence Scoring

**Purpose:** Assign a 0–3 score for each of 6 target skill domains based on *where* evidence appears in the profile.  
**Inputs:** Full candidate profile, compiled regex pattern lists.  
**Output:** Dict of `{domain_name: score}` + verbatim evidence snippets.  
**Logic:** If pattern matches in career description AND production signals also match → 3; if career description only → 2; if skills section only → 1; else → 0. Platform assessment score ≥70 adds 0.5 (capped at 3).  
**Plain English:** Reading a resume, you trust "I deployed this to production" more than a skills checkbox. The 0–3 scale quantifies that trust hierarchy.  
**Complexity:** O(D × P) where D = career description characters, P = number of patterns (~80). Fast regex on text strings.

---

## Algorithm 4: Consistency Score

**Purpose:** Detect keyword stuffers — people who claim expert/advanced in JD-relevant skills but whose career history doesn't support it.  
**Inputs:** Skill claims with proficiency levels, career history titles, career description text, skill durations.  
**Output:** Float [0.30, 1.00] — 1.0 means fully consistent, lower means more contradictions.  
**Formula:** `1.0 - 0.7 × (contradicted_claims / total_expert_claims)`  
**3 contradiction checks per expert/advanced claim:** (1) Career titles suggest completely different domain; (2) Skill duration < 12 months; (3) Skill term never appears in any career description (partial penalty: +0.5 contradiction).  
**Plain English:** If you claim "NLP: Expert" but your entire career is "BI Analyst at Deloitte" and you never mention NLP in any job, that's a flag. This score is applied as a multiplier — a 0.30 consistency score means all expert claims are contradicted.  
**Complexity:** O(S × T) where S = number of expert skills, T = career text length.

---

## Algorithm 5: 90-Day Plan Alignment

**Purpose:** Score how ready a candidate is to execute the JD's specific first-90-days plan.  
**Inputs:** Bucket A scores for retrieval (milestone 1), vector_db_hybrid + ltr_reranking (milestone 2), eval_framework (milestone 3); product_ratio from Bucket B.  
**Output:** Float [0, 1] alignment score → converted to **additive bonus** (not multiplier).  
**Formula:**  
```
m1 = retrieval_search / 3.0
m2 = max(vector_db_hybrid, ltr_reranking) / 3.0
m3 = eval_framework / 3.0

readiness = (m1 + m2 + m3) / 3.0
  +0.15 if all 3 milestones covered (capped at 1.0)
  −0.10 if only 1 covered (floored at 0.0)
  = 0.0 if none covered

alignment = 0.8 × readiness + 0.2 × product_ratio

ninety_day_bonus = 0.08 × alignment   → range: +0.000 to +0.080
```
**Plain English:** Covering all three milestones (retrieval audit, hybrid ranker, evaluation framework) with product-company experience makes you maximally ready to execute this specific role from day 1. This is a reward, not a gate — missing milestone evidence doesn't block a strong candidate.  
**Complexity:** O(1) — pure arithmetic on precomputed Bucket A/B values.

---

# 11. Configuration & Tuning

All tunable via `weights.yaml` — **no Python code changes needed.** Edit `weights.yaml` and re-run `rank.py`. See [variables.md](file:///d:/GitHub/evidence-rank/variables.md) for the full file with every number and its effect.

| Parameter | `weights.yaml` key | Default | Effect of Increasing | Effect of Decreasing |
|---|---|---|---|---|
| **Must-Have weight** | `phase4.must_have_weight` | **55%** | Technical skills dominate more | Behavioral/career signals matter more |
| **Product Builder weight** | `phase4.product_builder_weight` | **20%** | Shipping/startup background matters more | Technical keyword evidence dominates |
| **Career Quality weight** | `phase4.career_quality_weight` | **15%** | Domain trajectory matters more | Skills evidence dominates |
| **Nice-to-Have weight** | `phase4.nice_to_have_weight` | **10%** | LTR/LLM experience rewards higher | Must-have skills dominate even more |
| **Cross-Encoder weight** | `phase4.cross_encoder_weight` | **20%** (CE in Phase4) | Semantic match dominates | Handcrafted features dominate |
| **RRF k constant** | `retrieval.rrf_k` | **60** | High-rank positions matter less; mid-rankers favored | Top positions in any list are heavily rewarded |
| **FAISS top-k** | `retrieval.faiss_top_k` | **2,000** per vector | Broader recall, slower RRF | Tighter recall, faster, may miss edge candidates |
| **Ghost: days_inactive** | `ghost.days_inactive_threshold` | **365 days** | Fewer ghosts filtered | More candidates filtered |
| **Ghost: response_rate** | `ghost.response_rate_threshold` | **< 5%** | Almost no ghosts filtered | More ghosts filtered |
| **Availability (bad case)** | `availability.bad_mult` | **×0.70** | Inactive candidates penalized less | Inactive candidates penalized more |
| **Combined mult floor** | `floors.combined_floor` | **0.25** | Strong technical evidence shows through more | More extreme penalty stacking allowed |
| **Logistical floor** | `floors.logistical_floor` | **0.75** | Location/notice/seniority group less punishing | More range for operational signals |
| **LangChain-only penalty** | `penalties.langchain_only_mult` | **×0.45** | Near-disqualifying | Softer — LangChain profiles rank higher |
| **Research-only penalty** | `penalties.research_only_mult` | **×0.40** | Near-disqualifying | Researchers rank higher |
| **90-Day bonus scale** | `ninety_day.bonus_scale` | **0.08** (+0.00 to +0.08) | Stronger reward for milestone-ready candidates | 90-day coverage matters less |
| **Social boost cap** | `social.boost_cap` | **0.12** | Platform engagement signals matter more | Social proof less influential |
| **Recruiter saves boost** | `social.recruiter_saves_boost` | **+0.04** | "Popular candidate" signal stronger | Social proof less important |
| **Cross-encoder candidate count** | `retrieval.cross_encoder_top_n` | **Top 500** | Better reranking coverage | Faster offline precompute |

> [!TIP]
> The most impactful tuning knobs are: Must-Have weight (55%), the LangChain penalty (×0.45), and the combined_floor (0.25). NDCG@10 = 50% of the hackathon score — if top-10 quality seems wrong, adjust must-have weight or cross-encoder weight before anything else.

---

# 12. Critical Files

| File | Purpose | Why It Matters |
|---|---|---|
| [plan.md](file:///d:/GitHub/evidence-rank/plan.md) | Full production spec v3.3.0 — complete implementation blueprint | **The source of truth.** Every algorithm, formula, and threshold is defined here |
| [variables.md](file:///d:/GitHub/evidence-rank/variables.md) | Authoritative weight & variable reference; contains the full `weights.yaml` config | **Tuning source of truth.** Every number in the system, its stage, type (×/+/HARD), and the `weights.yaml` key to change it |
| [README.md](file:///d:/GitHub/evidence-rank/README.md) | Human-readable system overview and submission guide | What judges and reviewers read first |
| `weights.yaml` | Dynamic config file — all weights, multipliers, boosts, and thresholds | **Change any parameter here without touching Python code** |
| `rank.py` | Runtime entry point — the actual competition submission script | Must finish ≤5 min; no heavy imports (`torch`, `faiss`, `flagembedding`) allowed |
| `preprocess.py` | Offline pipeline runner — builds all artifacts | Runs once; may take 1–4 hours on CPU |
| `app.py` | HuggingFace Spaces demo (Gradio) | BM25-only heuristic path; no ML models; judges can interact with it |
| `src/retriever.py` | FAISS + BM25 + RRF fusion | At preprocess time: builds retrieval_scores.parquet. At rank time: loads it |
| `src/features.py` | Regex-based Bucket A/B/C extraction | The core "evidence vs claims" intelligence — all pattern lists live here |
| `src/scorer.py` | 4-component weighted formula (55/10/15/20) + CE merge | Phase 4 score — reads weights from `weights.yaml` |
| `src/reranker.py` | Cross-encoder offline inference | Runs bge-reranker-v2-m3 on top 500 offline; loads at rank time |
| `src/behavioral.py` | Availability, notice, location, seniority, penalty multipliers + social boost | Phase 5 — reads all multipliers/thresholds from `weights.yaml` |
| `src/explainer.py` | Reason generation with 90-day milestone framing | Phase 6 — zero LLM calls; pure string templates from evidence snippets |
| `artifacts/candidate_features.parquet` | Precomputed Bucket A/B/C for all retrieved candidates | The single most important artifact — everything Phase 4/5/6 reads from here |
| `artifacts/retrieval_scores.parquet` | RRF scores for top 5,000 | The gate into the scoring pipeline |
| `artifacts/cross_encoder_scores.parquet` | Cross-encoder raw logits and normalized scores for top 5,000 | Semantic tiebreaker |
| `artifacts/candidate_flags.parquet` | Honeypot/ghost/disqualifier flags | Hard exclusions — must be correct |
| `artifacts/faiss_index.bin` | FAISS IndexFlatIP of all 100K profiles | Used offline only for retrieval |
| `artifacts/jd_skills_vector.npy` + `jd_ideal_vector.npy` | Two JD embedding variants | Core of the dual-vector retrieval strategy |
| `metadata/validation_set.json` | 150–200 hand-labeled candidates | Used for offline quality checking — never used to overfit weights |
| `submission_metadata.yaml` | Team metadata for submission portal | Required for submission — team name BuriBuri, contacts, methodology |

---

# 13. What Happens To One Example Item

**Candidate:** "Arjun Mehta" — 7 years YoE, current title "ML Engineer", currently in Pune, at a product startup (e-commerce). Skills listed: "FAISS: Expert (36 months)", "Elasticsearch: Advanced (24 months)", "Python: Expert (72 months)". Career description at current job: *"Built a dense retrieval pipeline using FAISS and sentence-transformers serving 500K daily queries; implemented NDCG@10 evaluation for relevance benchmarking; shipped hybrid BM25+dense ranker to production, reducing p99 latency by 40%."* GitHub score: 75. Recruiter response rate: 82%. Notice period: 30 days. Last active: 15 days ago.

---

## Step 1: Phase 1 — Preprocessing (offline)

**Profile text built:**
> "ML Engineer [headline] [summary] FAISS Elasticsearch Python [career descriptions concatenated] skills: FAISS Elasticsearch Python..."

**Honeypot checks:**
- Skill "FAISS: 36 months" vs YoE 84 months → ✅ Fine (36 < 84+6)
- Expert skill duration > 0 → ✅ Fine
- No assessment scores to check → ✅ Fine
- No single role longer than total career → ✅ Fine
- **Result: is_honeypot = False**

**Ghost check:**
- Last active 15 days ago → fails condition (must be >365 days)
- **Result: is_ghost = False**

**Disqualifier tagging:**
- Not a consulting firm → consulting_only = False
- Has engineering titles → research_only = False
- Has NLP/retrieval terms, no CV/speech → wrong_domain = False
- Product company ratio: ~100% product → product_ratio = 0.98

**Saved to `candidate_flags.parquet`:** `{is_honeypot: false, is_ghost: false, product_ratio: 0.98, consulting_only: false, research_only: false, wrong_domain: false}`

---

## Step 2: Phase 1d — RRF Retrieval (offline)

**FAISS (skills vector):** "FAISS", "sentence-transformers", "dense retrieval", "NDCG", "hybrid search" all appear → likely rank ~30 in top 2,000.  
**FAISS (ideal vector):** "7 years", "product company", "shipped to production", "real users", "retrieval system" → likely rank ~25 in top 2,000.  
**BM25:** "FAISS", "NDCG", "BM25", "dense retrieval" exact-match keywords → likely rank ~15 in top 2,000.  

**RRF score:** `1/(60+30) + 1/(60+25) + 1/(60+15)` ≈ `0.0111 + 0.0118 + 0.0133` = **0.0362**  

This is a high RRF score — top 50 in the retrieved pool.

**Arjun is in the top 5,000. He enters the pipeline.**

---

## Step 3: Phase 1c — Feature Extraction (offline)

**Bucket A — Technical Skill Evidence:**

| Domain | Evidence | Score |
|---|---|---|
| `retrieval_search` | "FAISS and sentence-transformers", "dense retrieval pipeline" in career description + "serving 500K daily queries" (production) | **3** |
| `vector_db_hybrid` | "hybrid BM25+dense ranker", "FAISS" in description + production signals | **3** |
| `eval_framework` | "NDCG@10 evaluation for relevance benchmarking" in description | **2** (no explicit A/B test or online eval mention) |
| `ltr_reranking` | No "learning to rank", "LambdaMART", or "cross-encoder" mention | **0** |
| `llm_integration` | No LLM/fine-tuning mention | **0** |
| `python_coding` | Career description references Python patterns | **2** |

**Bucket B — Career Quality:**
- product_ratio = 0.98 ✅
- deploy_signal: "serving 500K daily queries", "p99 latency", "shipped" → 5+ production patterns → deploy_signal = 1.0
- experience_recency: Most recent role is ML Engineer with retrieval work → 1.0
- depth_signal: Retrieval patterns in current role → 1.0
- shipper_ratio: "shipped", "production", "serving", "deployed" → high shipper count vs zero researcher terms → shipper_ratio ≈ 1.0
- writing_signal: Description is ~200 chars → 1.0
- ownership_signal: "end-to-end" or similar → assume True (1.0)

**Product Builder Score:**
```
product_builder_score = 0.35 × 0.98 + 0.30 × 1.0 + 0.20 × 1.0 + 0.15 × 1.0
                      = 0.343 + 0.30 + 0.20 + 0.15 = 0.993
```

**Bucket C — JD Fit Gaps:**
- seniority_score: 7 years YoE → in sweet spot 5–9 → **1.00**
- title_velocity_flag: Normal tenure → False
- external_validation: github_score=75 → True
- code_stopped: ML Engineer, not Architect → False
- langchain_only_flag: No LangChain terms → False
- closed_source_flag: github_score > 0 → external_validation = True → False

**Consistency Score:**
- Checks "FAISS: Expert (36 months)": Career has FAISS in descriptions ✅; 36 months > 12 ✅; titles are ML Engineer ✅ → no contradictions
- "Python: Expert (72 months)": All checks pass → no contradictions
- **consistency_score = 1.0** (no contradictions)

**90-Day Alignment:**
- m1 (retrieval): 3/3 = 1.0
- m2 (vector/ltr): max(3/3, 0/3) = 1.0
- m3 (eval): 2/3 = 0.667
- readiness = (1.0 + 1.0 + 0.667) / 3 = 0.889; all 3 milestones covered → +0.15 → min(1.039, 1.0) = **1.0**
- alignment = 0.8 × 1.0 + 0.2 × 0.98 = **0.996**
- ninety_day_bonus = 0.08 × 0.996 = **+0.0797**

---

## Step 4: Phase 4 — Core Scoring + CE Merge (runtime)

**Must-Have Score:**
```
retrieval_ev  = 3/3 = 1.00  → 0.25 × 1.00 = 0.250
vectordb_ev   = 3/3 = 1.00  → 0.20 × 1.00 = 0.200
eval_ev       = 2/3 = 0.667 → 0.15 × 0.667 = 0.100
must_have_raw = 0.550
has_any_retrieval = True → no cap applies
must_have_score = 0.550 / 0.60 = 0.917
```

**Nice-to-Have Score:**
```
ltr_ev  = 0/3 = 0 → 0.07 × 0 = 0
llm_ev  = 0/3 = 0 → 0.03 × 0 = 0
nice_to_have_score = 0 / 0.10 = 0.0
```

**Career Quality Score:**
```
career_quality_raw = 0.08 × 1.0   (sys_experience_score — built retrieval with prod signals)
                   + 0.04 × 1.0   (experience_recency — current role is retrieval)
                   + 0.03 × 1.0   (depth_signal — retrieval across roles)
                   = 0.15
No consulting/research/wrong-domain flags
career_quality_score = 0.15 / 0.15 = 1.0
```

**Product Builder Score** (computed in Bucket B above): **0.993**

**Core Score:**
```
0.55 × 0.917 + 0.10 × 0.0 + 0.15 × 1.0 + 0.20 × 0.993
= 0.504 + 0.000 + 0.150 + 0.199
= 0.853
```

**Cross-Encoder Score:** Arjun is in top 500 by preliminary score. CE sees "dense retrieval pipeline FAISS NDCG hybrid ranker" paired with JD → likely score ≈ 0.85

**Phase 4 Final:**
```
0.80 × 0.853 + 0.20 × 0.85 = 0.682 + 0.170 = 0.852
```

---

## Step 5: Phase 5 — Behavioral Multipliers (runtime)

**Logistical group:**

| Multiplier | Value | Reason |
|---|---|---|
| Notice Period | ×1.00 | 30-day notice ≤30 days |
| Location | ×1.00 | Pune (primary preferred city) |
| Seniority | ×1.00 | 7 years — sweet spot |
| Writing Signal | ×1.00 | Avg description ≥150 chars |

```
logistical_mult = 1.00 × 1.00 × 1.00 × 1.00 = 1.00
logistical_mult = max(1.00, 0.75) = 1.00   ← floor not needed
```

**Soft Penalties:** ×1.00 (no flags triggered)  
**Consistency:** ×1.00 (consistency_score = 1.0)  
**Availability:** ×1.15 (active 15 days ago, 82% response rate, open to work)

```
combined_mult = 1.15 × 1.00 × 1.00 = 1.15
combined_mult = max(1.15, 0.25) = 1.15   ← floor not needed
```

**Additive bonuses:**
- ninety_day_bonus = **+0.0797** (computed above)
- Social Proof Boost:
  - github_score 75 > 60 → +0.03
  - Assume recruiter saves unknown → +0.00
  - **social_boost = +0.03**

**Final Score:**
```
final = 0.852 × 1.15 + 0.0797 + 0.03
      = 0.980 + 0.0797 + 0.03
      = 1.090
```

*(Score above 1.0 is possible when availability multiplier exceeds 1.0 and additive bonuses apply.)*

---

## Step 6: Phase 6 — Reason Generation (runtime)

**Best snippet:** "dense retrieval pipeline using FAISS and sentence-transformers" from `retrieval_search` bucket.  
**Best bucket:** `retrieval_search` → maps to "Weeks 4-8 hybrid ranker mandate"

**Generated reasoning:**
> "7-year ML Engineer; evidence: 'dense retrieval pipeline using FAISS and sentence-transformers serving 500K'; suited for Weeks 4-8 hybrid ranker mandate. Strong availability: 30-day notice, 82% recruiter response rate."

---

## Final Output for Arjun:

```csv
CAND_XXXXXXX,1,1.090,"7-year ML Engineer; evidence: 'dense retrieval pipeline using FAISS and sentence-transformers serving 500K'; suited for Weeks 4-8 hybrid ranker mandate. Strong availability: 30-day notice, 82% recruiter response rate."
```

**Arjun ranks #1 (or near it).** He has maximum technical evidence (3/3 retrieval, 3/3 vector DB), near-perfect product builder score (0.993), perfect career quality (1.0), full seniority, ideal location, ideal availability, zero penalties, and full 90-day milestone coverage. The only gap is zero LTR/LLM evidence (nice-to-haves worth 10%), but his 55% must-have score and 20% product builder score are near-ceiling.

---

*End of deep dive. Version matches plan.md v3.3.0 + weights.yaml Option A (confirmed).*
