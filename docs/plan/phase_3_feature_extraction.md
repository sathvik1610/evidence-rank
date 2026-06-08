## 7. Phase 3 - Candidate Feature Extraction

### 7.1 What This Phase Does

Phase 3 runs offline in `preprocess.py` through `run_phase_1c_features()`. It extracts recruiter-intent features for the retrieved pool and writes them to `artifacts/candidate_features.parquet`.

The implementation is in `src/features.py`. It must stay JD-contract driven: JD-specific skill patterns, seniority bands, disqualifier terms, location policy inputs, and penalty multipliers are loaded through `src/jd_intelligence.build_feature_contract()` from `metadata/JD_contract.yaml`.

Do not reintroduce local JD pattern arrays in `features.py`. The purpose of this phase is not to copy the JD into Python; it is to score candidate evidence according to the intent encoded in the YAML contract.

### 7.2 Contract Inputs

`build_feature_contract()` derives:

- `target_skills`: retrieval/search, vector/hybrid DB, evaluation, LTR/reranking, LLM integration, Python, distributed systems, and HR-tech exposure.
- career-quality patterns: retrieval, ranking, recommendation, production/scale, ownership, shipper/researcher vocabulary.
- fit-gap policy terms: stopped-coding titles, hands-on title exemptions, LangChain-wrapper terms, pre-LLM production terms, wrong-domain terms, research-title terms.
- seniority bands and disqualifier multipliers from `multipliers`.
- floor-exempt multiplier IDs from `metadata.multiplier_application.floor_exempt_multiplier_ids`.

The YAML remains the source of truth for recruiter intent. `weights.yaml` remains the source of truth for scoring weights and generic multipliers.

### 7.3 Bucket A - Skill Evidence

Each target skill bucket is scored from 0 to 3:

- `0`: no evidence.
- `1`: skill-list evidence only.
- `2`: career-history title or description evidence.
- `3`: career evidence plus localized production/scale evidence.

Verified Redrob skill assessments can add a small capped bonus when the skill is already evidenced.

Important guardrail: skill-list matches alone are intentionally weak. This prevents keyword-stuffed profiles from outranking candidates with actual shipped systems.

Bucket A searches career descriptions only. Job titles are excluded from the 2.0/3.0 evidence path because a title such as `Recommendation Systems Engineer` can otherwise inflate a weak chatbot or support role. Titles still contribute to broader career-quality/domain context where appropriate.

### 7.4 Bucket B - Career Quality

Bucket B computes:

- `product_ratio`
- `deploy_signal`
- `experience_recency`
- `depth_signal`
- `sys_experience_score`
- `shipper_ratio`
- `writing_signal`
- `ownership_signal`
- `product_builder_score`

`product_builder_score` is a first-class JD signal because the role wants a product engineer who has shipped ranking/search/recommendation systems, not a pure researcher or generic AI wrapper user.

Consulting-only, pure-research, and wrong-domain penalties are read from the YAML contract multipliers.

Current-role recency has a chatbot-dominant guard: if the most recent role is a customer-support chatbot / ticketing chatbot role with no ranking or recommendation-system evidence, `experience_recency` is set below neutral (`0.3`) instead of being treated as generic adjacent ML work. Vector DB or embedding usage inside a support chatbot is adjacent evidence, but it is not enough to satisfy the JD's ranking/search ownership intent. If the broader career has no ranking/recommendation evidence, chatbot-only production retrieval is capped at partial system evidence (`sys_experience_score <= 0.5`).

### 7.5 Bucket C - Fit Gaps

Bucket C computes soft gap flags:

- `title_velocity_flag`
- `consulting_flag`
- `external_validation`
- `code_stopped`
- `seniority_score`
- `langchain_only_flag`
- `keyword_stuffer_flag`
- `closed_source_flag`

`keyword_stuffer_flag` is raised when a candidate lists many target AI/search skills but lacks career evidence in core retrieval/vector/eval/ranking buckets and does not have a hands-on technical title. This is a penalty, not an automatic reject.

`code_stopped` uses YAML-derived stopped-coding titles plus YAML-derived hands-on-title exemptions, so hands-on architects and staff engineers are not blindly penalized.

### 7.6 90-Day Alignment

`compute_ninety_day_alignment()` measures readiness against the JD's first 90 days:

- Weeks 1-3: audit BM25/retrieval.
- Weeks 4-8: ship v2 hybrid ranker or LTR/reranking path.
- Weeks 9-12: build evaluation framework.

This is used as an additive bonus in Phase 5. It should reward readiness, not disqualify otherwise strong candidates.

### 7.7 Behavioral Extraction

`extract_behavioral()` passes Redrob behavioral fields through into the feature parquet with `beh_` prefixes. It does not compute final behavioral penalties; Phase 5 applies those at rank time.

### 7.8 Output Schema

`candidate_features.parquet` contains:

- candidate ID
- Bucket A evidence scores
- Bucket B career/product quality signals
- Bucket C fit-gap flags
- 90-day alignment
- behavioral pass-through fields
- `snippets_json` for Phase 6 explanations
- Phase 1f flags forwarded explicitly for Phase 4/5 consumption

Forwarding Phase 1f flags is required. Otherwise Phase 5 would silently default disqualifier flags to false.
