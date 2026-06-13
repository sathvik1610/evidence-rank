## 10. Phase 6 - Reason Generation

### 10.1 Purpose

Phase 6 runs after final scores and ranks are assigned. It must never affect candidate ranking.

The implementation is `src/explainer.py`. It generates short, evidence-safe explanations for the final CSV and a debug trace in `artifacts/ranking_debug.csv`.

Basic profile facts (`current_title`, `current_company`, `years_of_experience`, location, country, and behavior fields) are copied into `candidate_features.parquet` during Phase 3. The explainer uses these copied fields directly; it does not infer facts from model output.

### 10.2 Evidence Domains

The explainer ranks candidate strengths using the same feature columns produced by Phase 3:

- `retrieval_search`
- `sys_experience_score`
- `vector_db_hybrid`
- `ltr_reranking`
- `eval_framework`
- `product_builder_score`
- `python_coding`
- `llm_integration`

`ltr_reranking` must remain a first-class explanation domain because the JD explicitly values ranking systems, hybrid rankers, XGBoost/LTR, cross-encoders, and reranking.

### 10.3 Tone Rules

The lead sentence is based on the strongest domain:

- score `>= 3.0`: strong evidence, use snippet when available.
- score `>= 2.0`: profile-text evidence, but not necessarily production-localized. For ranks 1-30 this is phrased as positive relevance instead of a negative caveat, so the reasoning tone matches the high rank. For ranks 1-15, a concrete extracted snippet is used when available.
- score `> 0.0`: skills-list or weaker evidence.
- score `0.0`: limited direct evidence.

Support sentences mention a secondary capability for highly ranked candidates when the secondary signal is real. Support phrasing is varied deterministically by candidate id/rank to reduce repeated templates in Stage 4 review without introducing generative hallucination risk.

When available and material, the second sentence includes quantified hiring signals such as `30-day notice` or `82% recruiter response`. Positive signals are phrased as hiring fit; long notice periods, low response rates, location friction, no relocation, and weaker evaluation evidence are phrased as natural caveats.

Reasons should stay at 1-2 sentences. The first sentence starts with concrete profile facts and the primary evidence domain; the second sentence, when present, adds secondary JD alignment or a concern.

### 10.4 90-Day Milestone Framing

- Weeks 1-3: retrieval/search/system evidence.
- Weeks 4-8: vector DB, LTR/reranking, product ML, or LLM ranker evidence.
- Weeks 9-12: evaluation framework evidence.

This is only explanatory framing. The 90-day score itself is computed in Phase 3 and added in Phase 5.

### 10.5 Concerns

`get_largest_concern()` surfaces the highest-impact concern in this order:

- impossible or suspicious profile
- research-only background
- wrong-domain background
- LangChain-wrapper-only risk
- isolated template risk
- low career IR density with adjacent-career evidence
- target-skill or generic skill-duration contradiction
- consulting-heavy career
- code-stopped seniority risk
- title velocity
- no relocation outside preferred/welcome city bands

For ranks above 25, present a concern when one exists. For ranks above 70, acknowledge technical-depth gaps even when no specific flag exists.

Band-aware tone is deliberate:

- ranks 1-10: excellent fit, unless a major CE/logistics/manual-review caveat requires "Strong JD fit" tone
- ranks 11-25: strong fit
- ranks 26-50: solid JD-aligned profile
- ranks 51-75: relevant but partial fit
- ranks 76-100: borderline or adjacent fit with clear limitations

For exact recruiter/candidate retrieval candidates, the support sentence can explicitly mention recruiter/candidate matching, hybrid retrieval, ranking decisions, and evaluation when the corresponding runtime bonus flags are present. For top-ranked candidates with sustained `career_ir_density`, the support sentence may explicitly say the career pattern shows search/ranking/evaluation ownership. This is only emitted from extracted feature values and runtime calibration flags; the explainer still does not infer or invent evidence.

### 10.6 Debug Output

`rank.py` writes `artifacts/ranking_debug.csv` with:

- `candidate_id`
- `rank`
- `score`
- `final_score`
- `true_unclamped_final_score`
- `core_score`
- `ce_score`
- `reasoning`
- `concern`

The official output CSV contains only the required submission fields.
