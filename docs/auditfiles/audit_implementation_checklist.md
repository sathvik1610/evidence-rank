# Audit Implementation Checklist

Date: 2026-06-09

Source files reviewed:

- `docs/auditfiles/deep_candidate_audit.md`: 100% considered. Applied the career-IR-density/template-isolation findings, top-window overrank/underrank targets, logistics emphasis, and warning that exact ordering matters less than window membership. No section was skipped.
- `docs/auditfiles/top100_audit_report.md`: 100% considered. Applied the top-10 logistics critique, current/product-role distinction, evaluation ownership emphasis, explanation-quality notes, and caution against unsafe global floors. No section was skipped.
- `Resources/submission_spec.txt`: 100% considered. Confirmed CSV shape, top-100 requirement, no network/GPU runtime, reasoning quality, and metric weights.
- `Resources/job_description.txt`: 100% considered. Used the product shipping, ranking/retrieval/matching, evaluation framework, location, relocation, and notice-period intent.

## Checklist

| Item | Status | Resolution |
|---|---|---|
| Add Career IR Density instead of relying on one recycled role template | Done | `src/features.py` now emits career density, eval density, adjacent-career ratio, role counts, and `isolated_template_risk`. |
| Penalize isolated RAG/chatbot/churn/CV/MLOps template evidence conservatively | Done | Phase 4 and Phase 5 both apply `isolated_template_risk_mult`; repeated support-chatbot/RAG histories are explicitly adjacent risk. |
| Boost sustained product IR/matching language | Done | Product IR phrases such as matching layer, learned relevance, search/discovery, and personalization infrastructure count toward density. |
| Strengthen evaluation ownership | Done | Must-have eval weight increased, and strong retrieval + LTR + eval gets a larger Phase 4 bonus. |
| Reduce nice-to-have/template inflation | Done | Nice-to-have bucket weight reduced to 0.05 and product-builder weight increased to 0.25. |
| Strengthen logistics without hard-killing strong fits | Done | Location/no-relocation, bad notice/location combinations, and responsive-passive handling updated in Phase 5. |
| Avoid double-penalizing responsive passive candidates | Done | Not-open penalty is skipped when recruiter response rate is at least 0.60. |
| Improve explanations and concerns | Done | Explanations prioritize retrieval/eval/LTR evidence, mention sustained career ownership, and surface template/logistics concerns. |
| Keep honeypot conservative | Done | No new hard honeypot kill switch was added for template reuse; this remains a soft ranking penalty. |
| Validate final CSV | Done | `validate_submission.py team_BuriBuri.csv` passes. |
| Run targeted tests | Done | `tests/test_features.py tests/test_scorer.py tests/test_behavioral.py tests/test_explainer.py -q` passes. |
| Reject unsafe broad eval expansion | Done | Adding broad "offline metrics" eval phrases improved one candidate but degraded top-window membership, so that change was reverted. |

## Final Audit Diagnostics

Final top-15:

1. `CAND_0064326`
2. `CAND_0018499`
3. `CAND_0080766`
4. `CAND_0046525`
5. `CAND_0006567`
6. `CAND_0081846`
7. `CAND_0027691`
8. `CAND_0062247`
9. `CAND_0041669`
10. `CAND_0068811`
11. `CAND_0046064`
12. `CAND_0086022`
13. `CAND_0077337`
14. `CAND_0075574`
15. `CAND_0066999`

Manual top-10 set membership:

- Top 10: 7/10.
- Top 15: 10/10.

Major overrank corrections:

- `CAND_0098846`: rank 67, out of top 50.
- `CAND_0010541`: out of top 100.
- `CAND_0065195`: out of top 100.
- `CAND_0037566`: rank 96, out of top 50.
- `CAND_0005538`: rank 25, no longer top-10.
- `CAND_0002025`: rank 19, no longer top-10.

Top-50 pull-ups:

- `CAND_0083879`: rank 39.
- `CAND_0060054`: rank 30.
- `CAND_0011687`: rank 71. This remains unresolved. A broad eval-pattern fix moved it closer but harmed top-window quality, so the change was rejected as unsafe overfit.

## Changed Surface Area

Approximate implementation change allocation:

- `deep_candidate_audit.md`: 100% considered; about 80% translated into code/config/doc changes. Remaining exact-order desires were not forced when candidate evidence was not clearly worse.
- `top100_audit_report.md`: 100% considered; about 85% translated into code/config/doc changes. The broad current-role/eval phrase expansion was explicitly tested and rejected after degradation.
- Code/config changed: `src/features.py`, `src/scorer.py`, `src/behavioral.py`, `src/explainer.py`, `weights.yaml`, and tests.
- Artifacts regenerated: `artifacts/candidate_flags.parquet`, `artifacts/retrieval_scores.parquet`, `artifacts/candidate_features.parquet`, `artifacts/ranking_debug.csv`, and `team_BuriBuri.csv`.

