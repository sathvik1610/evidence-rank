## 9. Phase 5 - Behavioral Re-ranking + Penalization

### 9.1 Purpose

Phase 5 runs in `src/behavioral.py` after Phase 4 technical scoring. It applies reachability, logistics, seniority, writing-culture, social-proof, and JD fit-gap penalties.

Behavioral signals are late modifiers. Retrieval remains relevance-first so strong passive candidates are not lost early.

### 9.2 Contract-Driven Inputs

`behavioral.py` loads `build_feature_contract(constants.JD_CONTRACT_YAML)` and uses:

- preferred/welcome city bands from `location_tier_multiplier`
- explicit floor-exempt multiplier IDs from `metadata.multiplier_application`
- YAML-derived `keyword_stuffer_penalty`

Location city sets must not be hardcoded in `behavioral.py`. Update `metadata/JD_contract.yaml` when recruiter location intent changes.

### 9.3 Reachability

`reachability_multiplier()` applies:

- inactivity penalty
- not-open-to-work penalty
- low recruiter response penalty

The reference date is loaded from `artifacts/run_metadata.json` when available, so sample and full runs do not depend on today's clock.

### 9.4 Logistics

The logistics multiplier combines:

- notice period
- location
- seniority
- writing signal

The group is floor-capped through `behavioral.logistical_floor` in `weights.yaml`, because logistics should not fully erase strong technical fit.

### 9.5 Soft Penalties

`soft_penalties()` applies:

- contradiction consistency penalty
- target-skill duration contradiction penalty (by count)
- target-skill overclaim magnitude penalty (by maximum months overclaimed)
- title velocity
- code-stopped risk
- LangChain-only risk
- keyword-stuffer penalty
- remote-only preference
- research-only
- wrong-domain
- closed-source/no external validation
- weak adjacent-domain title with weak retrieval/vector/evaluation evidence
- current consulting firm with weak target/product evidence
- no relocation outside preferred/welcome cities
- >90-day notice plus no relocation outside preferred/welcome cities
- isolated template risk from weak career IR density or repeated adjacent RAG/chatbot/churn work
- low career IR density
- below-5-year YoE soft penalty
- >90-day notice period when evaluation evidence is absent

Generic penalty values live in `weights.yaml`. JD-specific penalty values that are explicitly represented as YAML multipliers should come from `metadata/JD_contract.yaml`.

The weak-IR thresholds, current-consulting product-builder threshold, adjacent-domain title terms, remote-work aliases, below-band YoE cutoffs, and long-notice cutoff are all tunable in `weights.yaml`; they are not hardcoded in `src/behavioral.py`.

Target-skill duration contradictions are intentionally narrower than generic skill-duration noise. Phase 1f only counts expert/advanced claims for retrieval, ranking, recommendation, search, vector DB, or reranking terms that exceed claimed YoE plus the configured buffer. The system penalizes these through two mechanisms: a base penalty for the count of contradictions, and a scaling magnitude penalty based on `max_target_skill_overclaim_months`. While minor overclaims remain light soft penalties because skill-duration metadata is noisy, massive overclaims (e.g., >12 or >24 months over actual YoE) trigger steeper severity modifiers. Hard impossible flags rely on absolute contradictions visible in the candidate JSONL, such as copied role histories.

Candidates below 5 years are not hard-filtered, because the JD explicitly says the 5-9 year range is a preference. They do receive a soft Phase 5 penalty so top ranks remain biased toward the author's intended senior IC profile unless the technical evidence is unusually strong.

Adjacent-domain and current-consulting penalties are also evidence-gated. They do not hard-filter a candidate merely for having a Computer Vision title or a current TCS/Infosys/Wipro-style employer. They apply only when the same candidate lacks project-level retrieval, vector, and evaluation evidence, or has weak product-builder evidence.

Responsive passive candidates are not double-penalized solely for `open_to_work_flag = false`: if recruiter response rate is at least `behavioral.passive_response_skip_threshold`, the not-open penalty is skipped. Missing country is treated as unknown logistics, not as abroad/no-relocation.

### 9.6 Floor Exemptions

The combined multiplier floor is useful for ordinary soft signals, but it must not rescue profiles that match explicit JD disqualifier intent.

`has_floor_exempt_penalty()` prevents `behavioral.combined_floor` from rescuing:

- consulting-heavy profiles
- pure research profiles
- wrong-domain CV/speech/robotics profiles without NLP/IR escape evidence
- LangChain-wrapper-only profiles
- keyword-stuffed profiles
- weak adjacent-domain profiles
- weak current-consulting profiles
- bad logistics combinations
- isolated template-risk profiles

The exemption list is read from `metadata.JD_contract.yaml`.

### 9.7 Social Proof

`social_proof_boost()` adds capped points for hard-to-fake platform signals:

- GitHub activity
- recent recruiter saves
- endorsements
- interview completion
- offer acceptance
- profile completeness
- LinkedIn connection
- fast response behavior

These boosts are additive and capped so they cannot replace technical fit.

The cap is intentionally low (`social_proof_max = 4.0`) because social proof is a tie-breaker. It should help distinguish two technically similar candidates, not move a weaker technical profile into the top 10.

### 9.8 Final Score

`compute_final_score()` uses:

```text
final = final_phase4_score * combined_multiplier
      + ninety_day_bonus
      + reachable_elite_plan_bonus
      + social_proof_boost
      + full_plan_bonus
      + same_project_full_or_partial_system_bonus
      + recruiter_workflow_bonus
      + passive_responsive_exact_fit_bonus
      + split_career_core_coverage_bonus
```

Honeypot or suspicious profiles short-circuit to the honeypot multiplier. Candidates with extreme penalty stacks can be forced to `0.0` through `behavioral.penalty_floor_zero`.

Strict JD disqualifiers and missing-must-have candidates receive near-zero hard-gate scores and fall naturally rather than being removed manually before ranking. Ranks are assigned deterministically by descending internal `true_unclamped_final_score`, then ascending `candidate_id`. The submitted CSV score is a fixed-scale monotonic display calibration derived from that internal score.
