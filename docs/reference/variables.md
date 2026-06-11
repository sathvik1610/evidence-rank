# Evidence-Rank Variable Reference

This file is the concise current tuning reference for the checked-in pipeline. The authoritative numeric values live in `weights.yaml`; structural paths and pool sizes live in `constants.py`.

## Runtime Gates

| Item | Current value | Source |
|---|---:|---|
| RRF precompute pool | 15,000 | `constants.RRF_PRECOMPUTE_TOPK` |
| Exact recall pool | 10,000 | `constants.EXACT_RECALL_TOPK` |
| Runtime RRF cutoff | 10,000 | `weights.yaml -> retrieval.runtime_top_k` |
| Phase 5 candidate pool after Phase 4/CE merge | 1,000 | `weights.yaml -> ranking.phase5_candidate_pool` |
| Cross-encoder precompute pool | 15,000 | `constants.CE_PRECOMPUTE_TOPK` |
| Cross-encoder merge | 68% handcrafted / 32% CE | `weights.yaml -> scoring.handcrafted_weight`, `scoring.cross_encoder_weight` |

## JD Hard Gates

Before final behavioral scoring and ranking, `rank.py` removes candidates with a hard JD exclusion reason and writes them to `artifacts/hard_disqualified_debug.csv`.

| Gate | Rule | Source |
|---|---|---|
| Missing true must-have | Zero core retrieval/ranking, vector/hybrid search, ranking evaluation, or Python after considering extracted features plus runtime raw-profile rescue signals; `must_have_raw_evidence_override` remains available for manual/raw review | `src.behavioral.hard_disqualification_reason` |
| JD disqualifier | impossible/suspicious/ghost, keyword-stuffer, LangChain-only, pure research, consulting-only, wrong-domain, title-chaser, or stopped-coding flags | `src.behavioral.hard_disqualification_reason` |
| Notice risk | `notice_period_days >= 120`; strong demotion, not hard gate | `weights.yaml -> behavioral.long_notice_exclusion_days` |
| Location risk | Known India location outside preferred/welcome cities with no relocation, or non-India with no relocation/work-practicality signal; strong demotion, not hard gate | `src.behavioral.has_location_risk` |

Notice/location risk weights remain documented for historical compatibility, but `compute_final_score` no longer short-circuits logistics to near-zero. Official Top 100 ranking is built from candidates that pass capability/trust gates, then logistics are applied as ordinary penalties and caveats.

## Core Score

`src/scorer.py` computes `core_score` on a 0-100 scale:

```text
core_score =
  0.55 * must_have_score
+ 0.05 * nice_to_have_score
+ 0.15 * career_quality_score
+ 0.25 * product_builder_score
```

Must-have inner weights:

| Signal | Weight |
|---|---:|
| retrieval/search evidence | 0.22 |
| vector DB / hybrid search | 0.16 |
| search/recsys/ranking system experience | 0.20 |
| evaluation framework | 0.17 |
| Python coding | 0.05 |

Nice-to-have inner weights:

| Signal | Weight |
|---|---:|
| learning-to-rank / reranking | 0.04 |
| LLM / RAG / fine-tuning | 0.03 |
| distributed systems / inference | 0.02 |
| HR-tech exposure | 0.01 |

Product-builder score is precomputed in `src/features.py` from product-company ratio, deployment language, shipper-vs-researcher language, and ownership evidence.

Audit-specific corrections now add small Phase 4 adjustments for strong retrieval + LTR + evaluation trifectas, sustained career IR density, and low-density/template-isolated IR evidence. These are configured in `weights.yaml` under `scoring.eval_trifecta_bonus`, `scoring.career_ir_density_*`, and `scoring.isolated_template_risk_mult`.

## Behavioral Score

For gate-passing candidates, `src/behavioral.py` computes:

```text
final_score =
  final_phase4_score * combined_multiplier
+ ninety_day_alignment * 5.0
+ social_proof_boost
```

The logistical group is floor-capped at `0.75`; the broader combined multiplier is floor-capped at `0.25` unless a floor-exempt JD disqualifier is present. These floors do not rescue hard-gated candidates because hard gates are applied before final ranking.

Key penalty and modifier families:

| Family | Examples | Source |
|---|---|---|
| Reachability | inactivity, not open to work, low recruiter response | `behavioral.*` |
| Logistics | notice period, location/relocation, seniority, writing signal | `behavioral.*`, `soft_penalties.*`; 120-day notice and impractical known location/no-relocation are strong demotions |
| Career IR density | sustained search/ranking/evaluation versus isolated chatbot/churn/CV templates | `src/features.py`, `scoring.*`, `soft_penalties.*` |
| JD disqualifiers | research-only, wrong domain, LangChain-only, keyword stuffing, consulting-only | hard gate first; residual soft penalties only apply to gate-passing borderline cases |
| Trust checks | contradiction counts, target-skill duration overclaims, honeypot score | `honeypot.*`, `soft_penalties.*` |
| Social proof | GitHub, recruiter saves, endorsements, interview/offer behavior, profile completeness, LinkedIn, fast response | `social_proof.*` |

## Redrob Signal Coverage

The current runtime uses 16 of 23 Redrob signals directly or through extracted features.

| Signal | Status | Where used |
|---|---|---|
| `profile_completeness_score` | Used | social proof |
| `signup_date` | Not used | omitted as low-signal for this JD |
| `last_active_date` | Used | reachability, ghost logic |
| `open_to_work_flag` | Used | reachability, ghost logic |
| `profile_views_received_30d` | Not used | omitted to avoid popularity bias |
| `applications_submitted_30d` | Used | ghost logic |
| `recruiter_response_rate` | Used | reachability, ghost logic, fast response |
| `avg_response_time_hours` | Used | fast response |
| `skill_assessment_scores` | Used | skill evidence bonus |
| `connection_count` | Not used | low role-specific signal |
| `endorsements_received` | Used | social proof |
| `notice_period_days` | Used | logistics; 120-day notice risk |
| `expected_salary_range_inr_lpa` | Not used | no salary target in JD/scoring spec |
| `preferred_work_mode` | Used | remote-preference penalty |
| `willing_to_relocate` | Used | location fit and location risk |
| `github_activity_score` | Used | social proof, external validation |
| `search_appearance_30d` | Not used | omitted to avoid popularity bias |
| `saved_by_recruiters_30d` | Used | social proof |
| `interview_completion_rate` | Used | social proof |
| `offer_acceptance_rate` | Used | social proof |
| `verified_email` | Not used | extracted but not scored |
| `verified_phone` | Not used | extracted but not scored |
| `linkedin_connected` | Used | social proof |

## Current Documentation Caveats

Historical backup and duplicate audit/reference notes have been pruned. Use `README.md`, this file, `weights.yaml`, `constants.py`, the current phase files, and `docs/auditfiles/team_buriburi_deep_audit.md` as the current reference set.
