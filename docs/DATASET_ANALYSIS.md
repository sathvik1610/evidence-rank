# Dataset Analysis And Assumptions

This note records how the project interprets the released Redrob hackathon data. It is meant to make the ranking assumptions defensible during manual review and interview, especially where the bundle warns about traps but does not fully define how every noisy pattern should be treated.

## Official Dataset Shape

The participant bundle describes the candidate pool as `100,000` profiles in `candidates.jsonl.gz` / `candidates.jsonl`. Each record follows `Resources/candidate_schema.json` and includes:

- static profile facts: title, company, location, country, years of experience, industry, current company size
- career history: company, title, dates, duration, current-role flag, industry, company size, description
- education, skills, certifications, and languages
- Redrob behavioral signals: activity, availability, response rates, notice period, recruiter interest, assessment scores, verification, GitHub activity, and offer/interview behavior

The official bundle also includes:

- `Resources/job_description.txt` - the only source of truth for JD intent
- `Resources/submission_spec.txt` - CSV format, runtime constraints, scoring, honeypot check, and manual reasoning checks
- `Resources/redrob_signals_doc.txt` - interpretation guide for the 23 Redrob behavioral fields
- `Resources/candidate_schema.json` - exact data schema
- `Resources/sample_candidates.json` - small profile sample for schema inspection
- `validate_submission.py` - local format validator

## Evaluation-Relevant Warnings

The official resources explicitly warn that the dataset contains traps:

- keyword stuffers
- plain-language Tier 5 candidates
- behavioral twins
- approximately 80 honeypots with subtly impossible profiles

The submission specification says honeypots are forced to relevance tier 0 and that a honeypot rate above 10% in the submitted top 100 causes disqualification at Stage 3. It also says reasoning is manually reviewed for specific facts, JD connection, honest concerns, no hallucination, variation, and rank consistency.

Because the hidden labels are not available, this project does not optimize to ground truth. It uses the JD, the schema, the behavioral-signal guide, and corpus-level audits to make general ranking choices.

## Corpus Audit Findings Used By The Ranker

Local audits of the released `100,000` profile pool found exact long role descriptions repeated across different companies at high frequency:

```text
candidates scanned: 100,000
candidates with 2+ exact duplicate long descriptions across companies: 33,786
candidates with 3+ exact duplicate long descriptions across companies: 3,164
```

This matters because a single duplicated retrieval/ranking paragraph can look like multiple independent systems if counted naively. At the same time, the pattern is too common to treat every affected candidate as fake or as one of the approximately 80 subtle honeypots.

The implemented assumption is therefore conservative:

- exact repeated descriptions are not automatic fraud
- the first occurrence can establish semantic evidence for retrieval, ranking, evaluation, or production-system work
- repeated copies are discounted for career-depth, role-count, and density signals
- structural evidence remains independent: employer, tenure, dates, current role, title, industry, company size, location, and Redrob behavior
- 3+ repeated descriptions remain a stronger trust concern, but still need surrounding evidence before being treated as a hard exclusion

## Behavioral Twins Interpretation

The official resources name "behavioral twins" but do not define a deterministic exclusion rule for them. The project therefore treats behavioral signals as recruiter-availability and trust modifiers, not as standalone identity/fraud labels.

This is important for two reasons:

- the JD says a perfect-on-paper candidate who has not logged in for months or has very low recruiter response is not practically available
- candidates with similar static resumes can differ materially in Redrob behavior, so behavioral signals help separate reachable candidates from paper-only matches

The ranker does not discard a candidate just because a static profile resembles another profile. It rewards or penalizes practical hireability through recency, response rate, open-to-work, recruiter interest, interview behavior, notice period, verification, and GitHub/activity signals.

## Skill Duration Reliability

Skill names and skill durations are useful but lower-trust than demonstrated work. The JD itself warns against AI keyword traps, and the schema contains self-reported skill-duration metadata that can be noisy in synthetic data.

The ranking logic therefore follows a reliability hierarchy:

1. demonstrated career/project evidence in role descriptions
2. current/recent role, product-company context, and production deployment signals
3. Redrob behavioral and availability signals
4. skills, certifications, endorsements, and assessment signals
5. skill duration metadata

For the JD's explicit Python requirement, the system accepts direct Python evidence first. It also accepts narrow, recruiter-defensible proxies when the full profile shows hands-on ML/search engineering: PyTorch, scikit-learn, MLflow, Kubeflow, or FAISS used in a shipped ML/search context. This is intentionally narrower than treating every vector database or AI tool as Python evidence; a bare Pinecone, Qdrant, Weaviate, or LangChain mention does not by itself prove the coding bar.

Minor duration inconsistencies reduce confidence where material, but do not automatically invalidate a profile when stronger career evidence supports the JD fit. Clear impossible contradictions and official honeypot-style patterns are still penalized.

Career-history text is the highest-trust signal, so lexical matching needs to preserve JD-equivalent phrasing. The ranker counts singular/compound evidence such as `embedding-based search`, `embedding ranker`, and `ranker variants` as retrieval/ranking career evidence. It does not treat generic `A/B testing` as evaluation evidence unless the same role is already anchored to search, ranking, retrieval, recommendation, or matching. This distinction is important in the synthetic corpus because experimentation language can appear in non-ML business roles, while the JD specifically values ranking-system evaluation and offline-to-online/A-B interpretation.

## 100K To 12K Preprocessing Reliability

The preprocessing stage is intentionally broad. It creates a candidate pool from multiple retrieval lanes instead of relying on one keyword search:

- dense semantic retrieval over JD-derived queries
- learned sparse retrieval
- BM25 lexical retrieval
- exact recall for explicit must-have and disqualifier-sensitive terms
- profile flags and feature extraction for trust, logistics, behavior, and career fit

The current artifact metadata records:

```text
candidate_count: 100000
reference_date: 2026-05-27
skip_embed: true
```

The broad pool is then reranked by `rank.py` under the submission constraints. The final ranking step is CPU-only, deterministic, no-network, and completes well under the 5-minute limit.

## Why These Assumptions Are Necessary

The JD says the right answer is not "find candidates whose skills section contains the most AI keywords." It asks for judgment about retrieval/ranking/evaluation depth, production history, product-engineering attitude, location/logistics, and behavioral reachability.

The assumptions above are designed to match that intent:

- avoid promoting keyword-stuffed or template-repeated evidence
- avoid falsely rejecting candidates for common synthetic-data artifacts
- keep strong plain-language builders in scope
- preserve practical hireability signals that a recruiter would care about
- keep every ranking choice reproducible from the released data and documented code

These assumptions are not hidden manual edits. They are implemented as general scoring and feature rules, tested where practical, and applied uniformly across candidates.
