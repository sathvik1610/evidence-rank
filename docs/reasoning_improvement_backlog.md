# Reasoning Improvement Backlog

Deferred task for later improvement of the `reasoning` column.

Current reasoning is safe and factual, but not fully polished. It passes validation and the local factuality audit, yet it can be made more evaluator-friendly.

## Target Improvement

Move from clipped snippet-based reasoning to cleaner grounded extractive reasoning:

- score full profile/career sentences, not only short regex windows
- prefer complete evidence sentences with JD-critical terms:
  - BM25
  - dense retrieval
  - FAISS / Pinecone / vector search
  - NDCG / MRR / recall@K
  - A/B testing
  - learning-to-rank
  - relevance judgments
  - scale, latency, QPS, corpus size
- avoid generic profile text such as "strong background in NLP"
- mention the top 2-3 actual ranking drivers in plain language
- include one honest concern when material: location, notice, low response, contradiction flags
- keep generation deterministic and auditable; no LLM calls during `rank.py`
- validate every claim against `candidates.jsonl`

## Current Quality Estimate

- Factuality: strong
- JD alignment: strong
- Specificity: good
- Human polish: still improvable

Main remaining issue: some snippets are clipped or awkward, for example partial phrases instead of full sentences.

## When To Do This

Only revisit after ranking is otherwise stable. Do not make broad scoring changes for this task.
