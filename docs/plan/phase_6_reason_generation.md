## 10. Phase 6 - Reason Generation

### 10.1 Purpose

Phase 6 runs after final scores and ranks are assigned. It must never affect candidate ranking.

The implementation is `src/explainer.py`. It generates short, evidence-safe explanations for the final CSV and a debug trace in `artifacts/ranking_debug.csv`.

Basic profile facts (`current_title`, `current_company`, `years_of_experience`, and location) are copied into `candidate_features.parquet` during Phase 3. The explainer uses these copied fields directly; it does not infer facts from model output.

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
- score `>= 2.0`: profile-text evidence, but not necessarily production-localized.
- score `> 0.0`: skills-list or weaker evidence.
- score `0.0`: limited direct evidence.

Support sentences mention a secondary capability for highly ranked candidates when the secondary signal is real. Otherwise they fall back to the JD's 90-day milestone framing.

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
- consulting-heavy career
- code-stopped seniority risk
- title velocity

For ranks above 30, present a concern when one exists. For ranks above 70, acknowledge technical-depth gaps even when no specific flag exists.

### 10.6 Debug Output

`rank.py` writes `artifacts/ranking_debug.csv` with:

- `candidate_id`
- `rank`
- `score`
- `core_score`
- `ce_score`
- `reasoning`
- `concern`

The official output CSV contains only the required submission fields.
