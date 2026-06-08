# Top-100 Ranking Audit

Author stance: skeptical JD author for `Resources/job_description.txt`. Evidence source: `team_BuriBuri.csv`, `candidates.jsonl`, `Resources/submission_spec.txt`, current `artifacts/candidate_flags.parquet`, and current code/config.

Scope note: I loaded and scored/scanned all top-100 profiles from `candidates.jsonl` for career history, skills, Redrob signals, education, and structural flags. I did not include a 100-row appendix because the requested deliverable sections A-G ask for detailed top-10 judgments, a top-50 scan, every clear misranking, systemic issues, and actionable corrections. Routine candidates with no clear correction are therefore omitted from section C by design; that omission means "no clear rank correction found," not "not reviewed."

## A. Top-10 Audit

| Rank | Candidate | Verdict | Strongest reason for | Strongest concern | Rank judgment |
|---:|---|---|---|---|---|
| 1 | `CAND_0046525` | Yes | Current role says: "Led the migration from keyword-based to embedding-based search across a 30M+ candidate corpus" and ran A/B tests with recruiter engagement impact. | Current company is `Genpact AI` / `AI Services`; profile has a 60-day notice. JD allows current consulting only if prior product exposure is real, but rank 1 should be a cleaner product-company fit. The CSV reasoning also uses a repeated snippet seen across several top candidates, so it does not prove this profile is distinctly stronger than rank 2. | Too high; shortlist, but not first call. |
| 2 | `CAND_0018499` | Strong Yes | Zomato role: "BM25 + dense retrieval (BGE embeddings, FAISS HNSW)" plus "NDCG, MRR, recall@K" and 50M+ queries/month. | Prefers `remote`, which is not fatal but less aligned to flexible hybrid. | Feels right; likely rank 1. |
| 3 | `CAND_0064326` | Strong Yes | Sarvam AI role: "Owned the ranking layer" and moved it to a learning-to-rank model with relevance labeling and train/eval workflow. | 45-day notice and not willing to relocate, though Gurgaon/Delhi NCR is acceptable. | Feels right. |
| 4 | `CAND_0046064` | Yes | Salesforce role: fine-tuned LLMs for candidate-JD matching and built an eval harness with ranking metrics. | Coimbatore and not willing to relocate; top-5 is aggressive for a location-mismatched candidate. | Slightly too high. |
| 5 | `CAND_0002025` | Yes | Apple role: shipped a marketplace recommendation system from offline experimentation to live A/B test in 5 months. | Trivandrum and not willing to relocate; current evidence is recommendation-heavy, less direct on recruiter/search matching. | Too high; belongs around 10-15. |
| 6 | `CAND_0086022` | Yes | Sarvam AI role has BM25+dense retrieval, BGE/FAISS, LLM reranking, and offline evaluation. | `Senior Applied Scientist` title plus target-skill duration contradictions in current flags; still no hard honeypot flag. | Slightly too high but shortlistable. |
| 7 | `CAND_0077337` | Strong Yes | Paytm role: "production recommendation system at a marketplace product" with behavioral re-ranking and A/B test. | Kochi with remote preference and 60-day notice; "Staff" title is not automatically bad but should be checked for title-chaser pattern. | Feels close; top 10 is defensible. |
| 8 | `CAND_0098846` | Maybe | upGrad role owns an e-commerce search ranking layer and relevance labeling/training/eval workflow. | Reasoning leads with RAG/Pinecone support text, while the current role evidence is stronger in LTR/e-commerce ranking; speech terms also appear. | Too high; rank 15-25. |
| 9 | `CAND_0005538` | Maybe | Adobe role has large-scale relevance infrastructure: "index refresh, query understanding, ranking calibration" over billions of documents. | Kolkata, not willing to relocate, 90-day notice; top-10 practical hireability is weak. | Too high; rank 20-35. |
| 10 | `CAND_0071974` | Yes | Netflix role: end-to-end recommendations pipeline with BGE embeddings, Pinecone retrieval, XGBoost LTR, and behavioral-signal integration. | Vizag, not willing to relocate; also CV/speech skill noise appears, though not a hard wrong-domain flag. | Slightly high but defensible top 15. |

Top-10 replacement candidates I would call earlier: `CAND_0081846` at rank 13 and `CAND_0006567` at rank 17. `CAND_0081846` has Razorpay product-company evidence for "BM25 + dense retrieval (BGE embeddings, FAISS HNSW)" and "NDCG, MRR, recall@K" plus 30-day notice and relocation. `CAND_0006567` is Noida-based and describes a matching-layer overhaul from heuristics to production ranking/personalization.

## B. Top-50 Scan

Major issues:

- Location/logistics penalties are too soft for top-10/top-50 ordering. Top 10 includes several candidates outside preferred/welcome cities and not willing to relocate (`CAND_0046064`, `CAND_0002025`, `CAND_0005538`, `CAND_0071974`).
- Current or partial consulting/services context is not discriminating enough. `CAND_0046525` ranks 1 despite current `Genpact AI` / `AI Services`; rank 46 has another Genpact AI candidate. The JD's escape clause allows prior product-company experience, but the ranking and explanation need to distinguish "currently services with clearly superior product-company history" from "currently services with only comparable evidence."
- The top 50 contains too many profiles whose reasoning says "less concrete deployment evidence" while JSONL actually has concrete ranking/eval text. That indicates evidence extraction or explanation selection is not using the strongest career-history field.
- Long notice is inconsistently handled. `CAND_0083307` at rank 50 and `CAND_0061265` at rank 43 have 120-day notice and should be below more reachable, similarly technical profiles.
- Weak current-role relevance appears in ranks 29, 32, 39, and 41: the current job descriptions are fraud/churn/RAG chatbot or adjacent ML work, while ranking/retrieval evidence comes from prior roles or skills.

Honeypot scan:

- No top-100 candidate is flagged `impossible_flag`, `suspicious_flag`, `is_ghost`, `consulting_only`, `research_only`, or hard `wrong_domain` in `artifacts/candidate_flags.parquet`.
- The current top 100 does contain 59 target-skill duration contradiction counts across candidates. These are not honeypots by themselves, but they are risk signals. Recheck at least `CAND_0075574` rank 27, `CAND_0065195` rank 30, `CAND_0030468` rank 47, `CAND_0030348` rank 99, and `CAND_0036863` rank 100 before final submission.
- I did not find a clear automatic honeypot in the top 100. Omission reason: current structural flags and manual spot checks did not prove an impossible profile, and the JD/spec warn against over-special-casing noisy duration metadata.

## C. Misranked Candidates

Clearly too high:

| Candidate | Tool rank | Suggested rank | Evidence / issue |
|---|---:|---:|---|
| `CAND_0046525` | 1 | 5-8 | Strong search profile, but current `Genpact AI` / `AI Services` and 60-day notice should lose to clean product companies like Zomato/Razorpay/Sarvam. |
| `CAND_0002025` | 5 | 12-18 | Strong recommendation evidence, but Trivandrum and `willing_to_relocate=false`; tool overweights technical/social signal. |
| `CAND_0098846` | 8 | 18-28 | JSONL has good LTR/eval evidence, but also speech-domain noise and less direct JD evidence than ranks 11, 13, 17, 18, and 19. |
| `CAND_0005538` | 9 | 25-35 | Strong retrieval infra text, but 90-day notice and no relocation from Kolkata make top-10 too high. |
| `CAND_0058688` | 32 | 55-75 | Current job is churn-prediction MLOps; retrieval/ranking comes mostly from adjacent/prior snippets and skills. |
| `CAND_0010541` | 29 | 60-80 | Current role says modeling work was secondary and production deployment was handled elsewhere; profile is closer to transition/adjacent ML than founding Senior AI Engineer. |
| `CAND_0061265` | 43 | 70-90 | 120-day notice and current rank should reflect high hiring friction despite good recommender/search evidence. |
| `CAND_0083307` | 50 | 80-100 | 120-day notice and current RAG chatbot evidence; too much support from skill/vector snippets. |

Clearly too low:

| Candidate | Tool rank | Suggested rank | Evidence / issue |
|---|---:|---:|---|
| `CAND_0081846` | 13 | 4-6 | Razorpay product role has "BM25 + dense retrieval (BGE embeddings, FAISS HNSW)", LLM reranker, LTR fallback, and NDCG/MRR/recall@K; 30-day notice and relocate. |
| `CAND_0006567` | 17 | 7-10 | Noida location; career text describes a matching-layer overhaul from heuristic system to learned relevance/personalization with online A/B framework. |
| `CAND_0027691` | 18 | 10-14 | Pune, Haptik NLP role, 15-day notice, and semantic search/FAISS/relevance evidence; stronger logistics than several current top-10 candidates. |
| `CAND_0041669` | 19 | 11-15 | Noida, CRED recommendation/search ranking layer, 8.0 years, active and product-company profile. |
| `CAND_0011687` | 38 | 15-25 | JSONL says "owned the offline-online evaluation harness — NDCG/MRR/recall calibrated to live A/B metrics" and current job has end-to-end ranking pipeline; 15-day notice. |
| `CAND_0083879` | 68 | 35-50 | Noida, 30-day notice, semantic search infrastructure from scratch with FAISS, and ranking/eval snippets. Reasoning undersells this as merely "mentions learning-to-rank". |
| `CAND_0060054` | 83 | 45-60 | Jaipur with relocation, 15-day notice, semantic search with FAISS/BGE and relevance judgments; not open-to-work is a concern but response rate is 86%. |

## D. Systemic Issues

Over-ranked profile types:

- Strong keyword/evidence profiles with weak practical logistics are too high. Location and notice penalties are capped by `behavioral.logistical_floor`, which preserves top-10 placement for candidates the JD author would call later.
- Services/current-consulting profiles with strong IR text are too high. The current `has_current_consulting_weak_ir` gate only applies when core IR/product evidence is weak, but the JD's preference is not just "avoid weak consulting"; it is "prefer product-company builders."
- Skill/vector/RAG evidence sometimes outranks direct product ranking/evaluation evidence. The report reasons for `CAND_0098846`, `CAND_0052328`, and lower-rank candidates often lead with RAG/Pinecone even when the better JD evidence is ranking/evaluation ownership.

Under-ranked profile types:

- Clean product-company search/recommendation engineers in preferred/welcome locations are underweighted when their evidence is plain-language or not top lexical snippets.
- Evaluation-heavy candidates are under-ranked. The JD explicitly says eval infrastructure is a first-90-days responsibility; `CAND_0011687` is a clear example.
- Reachable candidates with 15-30 day notice and acceptable location are not consistently lifted above technically similar but harder-to-hire candidates.

JD requirements not rewarded correctly:

- "Located in or willing to relocate to Noida or Pune" is treated as a soft afterthought in top ordering.
- "Product company, not pure services" is not strong enough in tie-breaks.
- "Hands-on evaluation frameworks" should be a top-tier discriminator, not only one subcomponent.

JD disqualifiers not enforced consistently:

- Current consulting/services is not a hard disqualifier and should not be, but it needs a top-rank demotion unless the product-company history is clearly superior.
- Wrong-domain skill noise is mostly handled by hard flags when core IR is absent; however, top candidates with substantial CV/speech noise still deserve trust penalties when the current role is adjacent.
- Long notice is only severe when eval is weak; for a founding-team role, >90 days should be a stronger practical penalty even with good eval evidence.

Reasoning sample finding:

Sampled ranks 1, 7, 13, 22, 31, 44, 57, 68, 83, and 100. The reasoning is not identical and usually references real fields, but it is still partly templated. Two main problems:

- It sometimes selects a weaker snippet than the best JSONL evidence. Example: `CAND_0081846` has full BM25/dense/LTR/eval current-role evidence, but the generated reason still uses the repeated "top-50, falling back..." phrasing.
- It understates lower-ranked strong candidates. `CAND_0083879` rank 68 is described as "mentions learning-to-rank", while JSONL shows semantic search infrastructure from scratch and product ranking/eval snippets.

Additional scoring risks to verify before changing weights:

- `ninety_day_alignment` and `social_proof_boost` are both additive and can contribute up to 10 total points. Check whether they compound on the same candidates before increasing eval or social-proof weights.
- `skill_assessment_scores` are the only platform-verified skill signal and should not be weakened without checking whether high assessment scores align with top manual picks.
- `open_to_work=false` plus high recruiter response may be double-penalized. Response rate already captures practical reachability for strong passive candidates.

## E. Weight And Signal Corrections Needed

| Issue | Correct behavior | Likely component |
|---|---|---|
| Location/logistics too weak in top 10 | For rank competition among high technical scores, preferred/welcome location and relocation should break ties strongly. Increase the differential between preferred-city/relocating candidates and no-relocation candidates outside preferred/welcome cities; reduce `logistical_floor` only carefully because the JD is flexible-hybrid, not daily office. | `weights.yaml`, `src/behavioral.py` |
| Current services/consulting profile can rank 1 | Do not hard-penalize current consulting when prior product-company evidence is real; that would contradict the JD. Instead, strengthen product-company ratio/current-product tie-breaking and make explanations surface the product-company history clearly. | `src/features.py`, `src/explainer.py`, `weights.yaml` |
| Evaluation ownership underweighted | Increase `eval_framework` contribution or add a trifecta/top-10 boost when retrieval + ranking + explicit NDCG/MRR/A/B evidence are all in current/recent roles. | `weights.yaml`, `src/scorer.py`, `src/features.py` |
| Long notice survives too high | Strengthen the >90-day penalty modestly, especially when combined with no relocation or weak current-role relevance. Avoid burying exceptional technical fits solely because buyout can partly solve notice. | `src/behavioral.py`, `weights.yaml` |
| Reason generator picks repetitive snippets | Choose snippets by strongest JD domain and current-role priority; prefer current/recent career descriptions over skills or generic summaries. | `src/explainer.py`, `src/features.py` |
| Lower ranks get generic weak language despite concrete evidence | Calibrate reason tone from evidence strength, not only rank. If a rank-68 candidate has project-level search/eval proof, say so and put the actual concern separately. | `src/explainer.py` |
| Target-skill duration contradictions too soft | Keep soft, but expose a clear trust concern in explanations when contradiction count >=2 or overclaim months >12. | `src/behavioral.py`, `src/explainer.py`, `weights.yaml` |
| Product-company preference not decisive enough | Increase `product_builder_weight` or add product-company tie-break for candidates with equivalent must-have scores. | `weights.yaml`, `src/scorer.py` |
| Current-role relevance is not prioritized | Score current/recent role evidence higher than old-role/skill evidence for the same keyword. | `src/features.py`, `preprocess.py` |
| Additive bonuses may double-count evidence | Audit `ninety_day_alignment` and `social_proof_boost` together before raising eval/social weights; they can add up to 10 points on a 100-point scale. | `src/behavioral.py`, `weights.yaml` |
| Open-to-work and response rate may double-penalize passive candidates | Consider reducing `not_open_mult` when recruiter response is high, because a responsive passive candidate is still reachable. | `src/behavioral.py`, `weights.yaml` |
| Verified skill assessments underused in audit | Keep assessment-score bonuses visible in candidate evidence and verify whether high scores in IR/search/recsys predict manual top picks. | `src/features.py`, `src/explainer.py` |

## F. Overall Verdict

1. The tool's top 10 does not fully match who I would call first. Wrong/high: `CAND_0046525`, `CAND_0002025`, `CAND_0098846`, `CAND_0005538`. Missing/too low: `CAND_0081846`, `CAND_0006567`, and probably `CAND_0027691`.
2. The tool reduces manual verification work on format and broad technical recall, but still creates ranking sanity-check work. A reviewer must inspect logistics, current company context, and explanation honesty for the top 50.
3. Highest-priority fix: strengthen top-rank tie-breaking around product-company/current-role evidence plus location/relocation. In practical terms, make a clean product-company candidate with current retrieval/ranking/eval evidence and reachable logistics beat a services/current-consulting or no-relocation candidate with similar keyword strength.

## G. Documentation Cleanup

Completed:

- Corrected README role wording, verified runtime wording, and Redrob signal coverage from 18/23 to 16/23.
- Updated `submission_metadata.yaml` with the actual sandbox link, root `candidates.jsonl` reproduce command, Windows local verification details, and `reproduction_tested: true`.
- Replaced stale `docs/reference/variables.md` with a concise current reference matching `weights.yaml`, `constants.py`, and code.
- Added a current-status note to `docs/reference/evidence_rank_deep_dive.md`.
- Updated `docs/plan/common_architecture.md` where it still said cross-encoder top 500 instead of the configured CE pool.
- Corrected current phase docs for `candidate_sparse_matrix.npz` and the current JD vector artifact names.
- Removed non-requested commentary from this report and folded useful issues back into sections D/E.

Not completed by design:

- Personal portal metadata remains `REQUIRED_BEFORE_PORTAL_UPLOAD`. I cannot truthfully fill names, emails, or phone numbers from repo evidence.
- Historical `docs/plan/plan_backup.md` remains stale because it is explicitly a backup. It should not be used as current implementation documentation.
- `test_candidate_sparse.npz` remains in `docs/plan/phase_0_jd_intelligence.md` because it is a scratch example filename, not a production artifact path.
