# Cross-Encoder Query Profile: Senior AI Engineer, Founding Team

## Role intent

This role is for a founding-team Senior AI Engineer who can own the intelligence layer of a talent/recruiting product. The core problem is not generic AI. The core problem is candidate-job matching: retrieval, search relevance, ranking, recommendation, reranking, and evaluation for real recruiter and candidate workflows.

The ideal candidate is a hands-on, product-minded AI/ML engineer who has shipped search, retrieval, ranking, recommendation, or matching systems to real users. They should combine deep technical judgment with a practical builder mindset. Prefer candidates who can ship an imperfect but useful ranking system quickly, measure it, learn from users, and improve it, rather than candidates who only discuss ideal research architectures.

This is a senior IC role that writes production code. Do not treat architecture-only, management-only, or research-only profiles as strong fits unless there is clear recent evidence of hands-on production engineering.

## Must-have evidence

Strong matches should show career-history evidence, not just skills-list claims, of several of the following:

- Production retrieval, search, ranking, recommendation, marketplace matching, candidate matching, job matching, personalization, or recommender-system work.
- Embedding-based retrieval, dense retrieval, sparse retrieval, hybrid search, BM25 plus vector search, semantic search, vector databases, approximate nearest neighbor search, or reranking systems.
- Experience with tools or systems such as FAISS, Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, OpenSearch, Lucene, Solr, BGE, E5, sentence-transformers, OpenAI embeddings, or similar. The exact tool matters less than production ownership.
- Strong Python and ML engineering evidence, preferably through production services, model pipelines, evaluation jobs, ranking pipelines, inference services, or backend systems supporting ML/search.
- Hands-on ownership of deployed systems used by real users, not only notebooks, demos, tutorials, internal prototypes, or coursework.

A candidate can be strong even without saying “RAG” or naming a specific vector database if their career history clearly shows they built search, recommendation, ranking, matching, or relevance systems.

## Strong positive signals

Give high credit to candidates who have shipped at least one end-to-end search, retrieval, ranking, recommendation, or matching system in a product environment.

Very strong evidence includes:

- Owning a ranking or retrieval system from design through deployment and iteration.
- Improving recruiter engagement, candidate matching quality, search relevance, feed quality, conversion, CTR, response rate, or other product metrics through ranking/search changes.
- Building hybrid retrieval systems, rerankers, learning-to-rank models, two-tower retrieval, cross-encoder reranking, recommendation models, personalization systems, marketplace ranking, or candidate/job matching systems.
- Making practical trade-offs between dense vs sparse retrieval, hybrid search, latency vs quality, fine-tuning vs prompting, offline metrics vs online behavior, and precision vs recall.
- Working in applied ML/AI roles at product companies, marketplaces, HR-tech, recruiting tech, search products, recommendation platforms, or user-facing ML systems.
- Showing senior judgment through clear ownership, technical trade-offs, production decisions, and measurable outcomes.

Open-source, papers, talks, or public technical writing are useful validation signals, but they are secondary to evidence of shipped production systems.

## Evaluation and experimentation signals

This role strongly values rigorous evaluation. Give significant credit to candidates who have designed or used evaluation frameworks for ranking, retrieval, recommendation, or search systems.

Positive evidence includes:

- NDCG, MRR, MAP, Recall@K, Precision@K, hit rate, relevance judgments, golden datasets, offline benchmarks, holdout sets, regression tests, or evaluation harnesses.
- Offline-to-online metric correlation, A/B testing, live experiments, recruiter-feedback loops, click/reply/engagement calibration, or online quality monitoring.
- Comparing baselines, measuring quality lift, debugging ranking regressions, interpreting A/B results, or improving model quality based on user behavior.
- Understanding that a ranking system is not good just because the model is modern; it must improve user-facing outcomes.

Do not reward generic A/B testing, marketing experimentation, or growth metrics unless they are tied to search, retrieval, ranking, recommendation, matching, or ML product quality.

## Production ownership signals

Production ownership is central. Prefer candidates who have operated systems under real constraints.

Strong evidence includes:

- Handling real users, live traffic, production latency, p95/p99 latency, model serving, online serving, index refresh, index rebuilds, embedding drift, model drift, data drift, retrieval-quality regressions, monitoring, incidents, alerts, rollbacks, or quality regressions.
- Owning not only model development but also deployment, monitoring, evaluation, iteration, and reliability.
- Building systems that needed to stay useful as data changed, users behaved unexpectedly, or product requirements evolved.
- Working closely with product, recruiter-experience, search, marketplace, or business teams to improve actual workflow outcomes.

A profile that only says “built a chatbot,” “used LangChain,” or “implemented RAG” is weak unless it also shows production usage, retrieval/ranking depth, evaluation rigor, and ownership of quality.

## Negative signals / profiles to penalize

Penalize profiles that are mostly keyword matches without career evidence.

Strong negative signals:

- Skills-only AI profiles where the career history does not show relevant search, retrieval, ranking, recommendation, matching, or production ML work.
- Shallow LangChain, OpenAI API, prompt-engineering, or chatbot-only experience, especially if it is recent and not backed by earlier production ML/search experience.
- Pure research, academic lab, PhD-only, or research-only profiles without production deployment or real-user product relevance.
- Generic ML, NLP, data science, computer vision, speech, or robotics profiles without meaningful retrieval, ranking, recommendation, search, or matching evidence.
- Framework enthusiasts whose work is mostly tutorials, demos, wrappers, or “built X using hot framework” posts rather than owned systems.
- Senior architects, managers, directors, or tech leads who no longer write production code and lack recent hands-on engineering evidence.
- Candidates whose entire career is in services/consulting with no product-company or product-like ownership, unless they clearly owned relevant production ML/search systems.
- Title-chasing patterns with very short tenures and no evidence of long-term ownership.
- Closed-source-only senior profiles with no way to validate technical thinking, unless the career history itself gives strong, concrete production evidence.

Do not over-penalize candidates just because they lack exact keywords. Penalize only when the profile lacks substantive evidence.

## Do not over-require exact keywords

Do not require exact JD keywords if the underlying work clearly matches. A candidate may be strong if they describe marketplace relevance, search quality, candidate-job matching, feed ranking, personalization, recommendation quality, talent matching, or relevance optimization without using the exact words “vector search” or “learning to rank.”

A candidate from search, recommendation, ads ranking, marketplace ranking, feed ranking, candidate matching, job matching, or product ML can be highly relevant even if they do not use the same words as the JD.

## How to judge evidence

Career-history evidence is more important than skills-list evidence. A skills list can support the match, but it should not carry the match by itself.

Prefer specific, outcome-oriented statements over broad claims. For example, “owned hybrid retrieval and reranking for marketplace search, improving NDCG and recruiter reply rate” is much stronger than “skilled in AI, RAG, LangChain, Pinecone, OpenAI.”

Look for what the candidate actually did:

- Did they build or only use?
- Was it deployed or only prototyped?
- Was it evaluated rigorously or only claimed to work?
- Was it used by real users?
- Did they own quality after launch?
- Did they make ranking/retrieval trade-offs?
- Did they improve measurable product outcomes?

## Logistics and availability

Technical fit is primary. Logistics are secondary but still useful for final ranking.

Prefer candidates in or willing to relocate to Pune, Noida, Delhi NCR, Mumbai, or Hyderabad. Shorter notice periods and clear job-market activity are positive signals. Candidates who are inactive, unreachable, have very low recruiter response rate, or show no recent job-market signal should be downweighted even if technically strong.

Do not let location or notice period outweigh strong technical evidence, but use them to break ties among similarly strong candidates.

## Tie-breaker priority

When two candidates are similar, prefer the one with stronger career-history evidence of production retrieval/ranking/matching, stronger evaluation practice, and clearer ownership of real-user outcomes. Use logistics, activity, notice period, and location only as secondary tie-breakers.

## Final scoring mindset for reranking

The best candidate is not the one with the most AI keywords. The best candidate is the one whose career history shows they can own a production retrieval/ranking/matching system, evaluate it properly, improve real user outcomes, and keep it reliable in production.

Rank highest:

1. Product-minded senior AI/ML engineers with shipped search/ranking/recommendation/matching systems.
2. Engineers with strong retrieval plus evaluation plus production ownership.
3. Candidates who show measurable product impact and practical trade-off judgment.
4. Candidates with relevant marketplace, HR-tech, recruiting, search, or recommendation experience.

Rank lower:

1. Keyword-heavy profiles with little career evidence.
2. Demo-only RAG/chatbot/LangChain profiles.
3. Pure researchers without production relevance.
4. Generic ML/CV/NLP profiles without retrieval/ranking/recommendation evidence.
5. Architecture-only senior profiles without recent hands-on coding.
6. Inactive or unreachable candidates, especially when there are similarly strong active candidates.

Be strict but fair. Reward real shipped systems and concrete evidence. Do not require exact terminology when the underlying experience clearly matches the role.
