# Intelligent Candidate Discovery & Ranking Engine
## Production System Specification — Version 2.0.0 (Final)

**Project:** Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge  
**Role Being Ranked For:** Senior AI Engineer, Founding Team, Redrob AI  
**Dataset:** 100,000 candidates in `candidates.jsonl` or `candidates.jsonl.gz`  
**Output:** `submission.csv` — top 100 candidates, ranked best-fit first  
**Hard Constraints:** 5-minute wall-clock execution, CPU-only, 16 GB RAM, 5 GB disk/intermediate state, zero network calls during ranking

---

## Table of Contents

1. Problem Understanding & Design Philosophy  
2. System Architecture Overview  
3. Repository Layout  
4. Phase 0 — Local Evaluation Harness  
5. Phase 1 — Streaming Ingestion & Recall Filter  
6. Phase 2 — Tri-Vector Retrieval Funnel  
7. Phase 3 — Offline Feature Factory  
8. Phase 4 — Weight Optimization Loop  
9. Phase 5 — Runtime Scoring Engine (rank.py)  
10. Phase 6 — Dynamic Reasoning Generator  
11. Phase 7 — Submission Compliance & Output  
12. Data Field Reference  
13. Known Traps & How Each Module Handles Them  
14. Dependency Declarations  

---

## 1. Problem Understanding & Design Philosophy

### 1.1 What the hackathon is actually testing

The challenge is not asking you to find candidates who contain the most AI keywords. It is asking you to reason about candidates the way a skilled recruiter would. The job description for Senior AI Engineer at Redrob AI is unusually explicit about what it wants and, more importantly, what it does not want. The dataset has been deliberately constructed with traps that punish naive keyword matching.

The scoring weights confirm this: NDCG@10 carries 50% of the total score. Your top 10 picks matter more than everything else combined. Getting 10 genuinely strong candidates into the top 10 is more valuable than having a well-ranked set of 50 mediocre ones.

### 1.2 What the job description actually means

Read the JD not as a checklist but as a personality profile of the company's hiring logic. The JD is written in honest, plain language — decode it as a human recruiter would, not as a keyword matcher.

**Hard requirements the system must operationalize:**

- Production experience with embeddings-based retrieval systems deployed to real users. The key word is production. Side projects, tutorials, and Kaggle notebooks do not count. The career history descriptions must contain evidence of deployment at scale — specifically things like "handled embedding drift," "index refresh," "retrieval-quality regression in production."
- Production experience with vector databases or hybrid search. Same logic. It does not matter which specific database they used (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). It matters that they operated one in a real system with real users.
- Strong Python with evidence of code quality, not just familiarity.
- Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation. This is a strong positive signal. Most candidates will not have it.

**Hard disqualifiers the system must penalize heavily:**

- **Pure research background with no production deployment.** These candidates will have strong academic credentials, possibly good skills listings, but their career history descriptions will lack deployment evidence. Titles like "Research Scientist" at universities or research-only labs (IIT, IISc, TIFR, DeepMind Research, etc.) with zero "Engineer" or "Developer" roles in career_history. Apply a 0.50 research_penalty multiplier.
- **"AI experience" consisting primarily of LangChain wrappers calling hosted APIs with under 12 months of this kind of work and no pre-LLM ML background.** These candidates will have LangChain, OpenAI, LlamaIndex in their skills but shallow career history descriptions with no retrieval or ranking evidence pre-2022.
- **Candidates whose ENTIRE career is at consulting firms.** TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, Mphasis, Hexaware, Mindtree, LTIMindtree, Capgemini, Zensar, KPIT, Birlasoft, Persistent Systems. CRITICAL JD NUANCE: If the candidate is currently at one of these firms but has prior product-company experience, that is explicitly acceptable per the JD. The penalty applies only when the ENTIRE career history (all roles) maps to consulting firms with zero product company exposure. Product ratio = 0.0 is the hard penalty trigger.
- **Computer vision, speech, or robotics specialists without NLP or IR exposure.** These candidates will have strong skills in OpenCV, YOLO, ASR, TTS, image classification, but shallow or absent retrieval and ranking history. A CV specialist who ALSO has NLP/search work in their career history is fine — the disqualifier is when CV/Speech is their ONLY domain.
- **Title-chasers.** The JD explicitly says: "If your career trajectory shows you optimizing for Senior→Staff→Principal titles by switching companies every 1.5 years, we're not a fit." Detectable via: avg_tenure_per_role < 18 months AND num_roles >= 3. Apply a 0.85 title_chaser_modifier.
- **Senior architects who stopped writing production code in 18+ months.** Detectable via: most_recent_role_title contains "Architect", "VP", "CTO", "Head of Engineering", "Director" AND yoe > 8. Apply a 0.85 architecture_role_modifier.
- **Framework-demo enthusiasts without systems evidence.** The JD explicitly rejects candidates whose AI signal is mostly LangChain/OpenAI/LlamaIndex demos, tutorials, or blog posts rather than retrieval/ranking systems. Detectable via framework keywords plus tutorial/demo wording, with low or zero retrieval/ranking/evaluation/production evidence. Apply a 0.80 framework_demo_modifier.
- **Closed-source-only proprietary backgrounds with no external validation.** The JD says 5+ years entirely on closed-source proprietary systems without papers, talks, or open-source makes the candidate hard to trust. Detectable via yoe >= 5, no GitHub activity, no external-validation terms in career text, and no open-source/publication/talk signals. Apply a 0.90 external_validation_modifier.

**Nice-to-have signals that earn a bonus multiplier:**

- LLM fine-tuning experience (LoRA, QLoRA, PEFT) — if present in skills AND corroborated in career descriptions, apply a 1.05 bonus.
- Learning-to-rank models (XGBoost-based LTR, LambdaMART, neural LTR) — same as above.
- Open-source contributions in AI/ML — detectable via github_activity_score > 50.
- HR-tech, recruiting-tech, talent intelligence, marketplace, or matching-product exposure — small positive bonus because the JD names this as useful but not required.
- Distributed systems, large-scale inference, latency, serving, or optimization experience — small positive bonus when supported by career descriptions.

**The JD's ideal candidate profile for the scoring formula:**

- 6-8 years total experience (soft window — 4 years acceptable if other signals are very strong, per JD). YoE is NOT a hard cutoff.
- 4-5 of those years at product companies in applied ML roles (not pure services).
- Has shipped at least one end-to-end ranking, search, or recommendation system at real scale.
- Located in or willing to relocate to Pune or Noida. Tier-1 Indian city candidates willing to relocate are explicitly welcomed.
- Active on the platform — recently logged in, responds to recruiters, notice period under 30 days is strongly preferred.

### 1.3 What the behavioral signals actually mean

The 23 redrob_signals fields are not profile quality metrics. They are availability and reachability metrics. A candidate with a perfect skills profile and a 3% recruiter response rate who last logged in 8 months ago is, for practical hiring purposes, not a real option. The signals should be used as a multiplicative modifier on top of the skill-and-experience score, not as a primary ranking signal.

**The critical signals for this role (used in S_Engagement score):**
- `last_active_date` — recency of platform activity. >180 days inactive = practically unreachable.
- `open_to_work_flag` — explicit availability signal. True is a major positive.
- `recruiter_response_rate` — reachability. <0.30 = unreachable even if technically excellent.
- `notice_period_days` — time to hire. JD explicitly prefers sub-30 days. 90+ days is a significant concern.
- `interview_completion_rate` — reliability signal. <0.50 = ghost risk.
- `applications_submitted_30d` — if > 0, candidate is actively looking right now.

**Secondary signals (used as modifiers):**
- `github_activity_score` — for an AI engineer role, active GitHub (>0) is a positive signal; -1 means no GitHub linked (neutral, not negative); >50 earns a bonus multiplier.
- `avg_response_time_hours` — very high (>72h) is a mild reachability concern.
- `profile_views_received_30d` / `search_appearance_30d` — recruiter visibility signals. Use lightly; high visibility can mean market validation, but it can also reflect generic profile exposure.
- `saved_by_recruiters_30d` — market validation; other recruiters are also interested in this person.
- `profile_completeness_score` — <50 indicates a disengaged or abandoned profile.
- `offer_acceptance_rate` — -1 means no history (treat neutral); <0.30 means likely to decline offers.
- `verified_email` / `verified_phone` / `linkedin_connected` — basic platform legitimacy.
- `endorsements_received` — overall social proof; use only as a light credibility modifier because per-skill endorsements already feed Skill Trust.

**Signals NOT used in ranking (no JD basis):**
- `expected_salary_range_inr_lpa` — JD mentions no salary-based filtering.
- `signup_date` — not a quality signal.
- `connection_count` — too noisy to use directly.

### 1.4 The trap categories and how each one fails a naive system

**Keyword stuffers.** These candidates list every AI keyword in their skills array but their career history descriptions don't back it up. A naive embedding system ranks them highly because their skill text is dense with target tokens. The Skill Trust Scorer catches them by requiring corroboration from career descriptions and checking duration and endorsements.

**Honeypots with impossible timelines.** These candidates have structural impossibilities: a skill listed with duration_months greater than their entire career length, expert proficiency in a skill with assessment score below 40. These must be HARD EXCLUDED (score = 0.0) — not just soft-penalized. See Section 7.2 for the hard gate logic.

**Plain-language genuine fits.** These are real strong candidates who described their work without using the fashionable keywords — they might say "built a recommendation system" rather than "deployed dense retrieval with FAISS." A system that only does keyword matching on skills text will miss them. The career semantic retriever and regex evidence extractor catch them by reading the full profile and career description text.

**Behavioral twins.** Two candidates with near-identical skill profiles and career histories, but one hasn't logged in for 6 months and has a 5% response rate while the other is actively seeking. They should not rank equally. The platform engagement score separates them.

**Consulting-but-nuanced candidates.** A candidate currently at TCS but who spent 5 years at Swiggy or Razorpay before is NOT a consulting-only candidate. The product_ratio formula correctly captures this because the consulting time is only a portion of their career. Do not disqualify them.

**Pure-CV/Speech engineers.** Strong technical background but entirely wrong domain. Career descriptions full of "object detection", "speech recognition", "robotics" with zero retrieval or NLP work. They will score low on S_Tech and S_Semantic naturally — the embedding similarity to the JD vector will be low.

---

## 2. System Architecture Overview

The system is divided into two strictly separated execution environments:

**The Offline Factory** (no time limit, run once before submission):
- Streams all 100,000 candidates and filters to a high-signal pool of 40,000-50,000
- Runs tri-vector retrieval to narrow to a rich-evidence operational pool of approximately 2,500-4,000 candidates
- Builds a full candidate feature store for all 100,000 candidate IDs so `rank.py` never has to extract features for the full pool at runtime
- Runs expensive semantic/evidence extraction only where useful; low-signal candidates receive explicit zero/default technical features
- Serializes all runtime-ready scalar features to `artifacts/candidate_features.parquet`
- Serializes optimized weights and any small lookup arrays needed by `rank.py`
- Optimizes scoring weights against the local validation set
- Saves optimized weights to `artifacts/optimized_weights.json`
- Saves deterministic runtime metadata, including `reference_date = max(last_active_date)`, to `artifacts/run_metadata.json`

**The Runtime Engine** (must complete in under 5 minutes on CPU, no GPU, no network, <=5 GB disk/intermediate state):
- `rank.py` is the only script that runs at evaluation time
- Loads precomputed scalar features from `artifacts/`
- Runs the vectorized 7-component scoring formula in NumPy
- Applies soft modifiers
- Generates reasoning strings
- Validates and writes the final CSV

The two environments share no code path during ranking. The offline factory can use any compute it needs. The runtime engine touches nothing that requires network or heavy computation.

During ranking, external services are forbidden. `rank.py` must not call OpenAI, Anthropic, Cohere, Gemini, hosted embedding APIs, hosted LLM APIs, package downloads, or any other network endpoint. Any model or embedding dependency needed by `rank.py` must already be local.

---

## 3. Repository Layout

Use this repository layout unless implementation details force a minor variation. The official requirement is not an exact directory tree; it is a clean, complete repo with a single reproducible ranking command, full source code, required artifacts or artifact-generation scripts, pinned dependencies, metadata, and a working sandbox/demo link.

```
├── rank.py
├── preprocess.py
├── train_weights.py
├── requirements.txt
├── submission_metadata.yaml
├── README.md
├── src/
│   ├── __init__.py
│   ├── indexer.py
│   ├── features.py
│   ├── inference.py
│   ├── funnel.py
│   └── explainer.py
├── metadata/
│   └── validation_set.json
└── artifacts/
    ├── candidate_features.parquet
    ├── jd_query_embedding.npy              # optional; only needed if semantic scores are not stored as scalars
    ├── embedding_candidate_ids.npy         # optional; only needed if runtime uses embeddings
    ├── optimized_weights.json
    └── run_metadata.json
```

**What each file does:**

`rank.py` — The single entry point for ranking. Takes `--candidates` and `--out` as CLI arguments. Loads precomputed artifacts, scores candidates, generates reasoning, writes the output CSV. Must complete in under 5 minutes.

`preprocess.py` — Runs the full offline pipeline: streaming filter → retrieval funnel → feature factory → embedding generation. Writes all artifacts. Can take hours; this is not constrained by the 5-minute rule.

`train_weights.py` — Loads the validation set, runs the scoring formula with variable weights, optimizes using SciPy Nelder-Mead, saves the result to `artifacts/optimized_weights.json`.

`src/indexer.py` — The streaming token filter. Reads `candidates.jsonl` or `candidates.jsonl.gz` line by line and writes `artifacts/high_signal_pool.jsonl`.

`src/features.py` — Contains the Skill Trust Score formula, the Profile Consistency Score formula, and the Product Company Classifier.

`src/inference.py` — Runs the regex-first evidence extractor over the operational pool.

`src/funnel.py` — Runs the three retrievers and returns the union set of candidate IDs.

`src/explainer.py` — Contains the dynamic reasoning generator function.

---

## 4. Phase 0 — Local Evaluation Harness

### 4.1 Why this must be built first

Without a local scoring harness, every decision about weights, thresholds, and feature importance is a guess. The harness gives you a feedback loop: change a formula, run the harness, see if NDCG@10 went up or down. This is the only way to make evidence-based decisions before the final submission.

### 4.2 The validation set — `metadata/validation_set.json`

Manually inspect 150-200 candidates from the actual dataset. Distribute them across four tiers:

**Tier 3 (Core Fit) — target 40 candidates:**  
These are candidates who should rank in the top 20-30 of your final submission. The criteria for labeling a candidate Tier 3:
- years_of_experience between 5 and 10
- Career history at product companies (not exclusively consulting firms)
- Career descriptions that contain evidence of building retrieval systems, ranking systems, or recommendation systems at production scale
- Skills that include at least some of: embeddings, vector databases, NLP, Python, ML evaluation metrics
- Behavioral signals showing active platform engagement

**Tier 2 (Borderline Fit) — target 60 candidates:**  
Strong engineers who are adjacent but not ideal. Examples:
- Backend engineers with strong Python and infrastructure but no retrieval/ranking history
- ML engineers with solid experience but primarily in computer vision or speech
- AI engineers with good credentials but consulting-only backgrounds
- Candidates who would be strong fits but have notice periods above 90 days or very low recruiter response rates

**Tier 1 (Noise) — target 30 candidates:**  
Clearly misaligned profiles. HR managers, accountants, civil engineers, operations managers, content writers. These exist in the dataset and a naive keyword system may surface them.

**Tier 0 (Honeypots) — target 20 candidates:**  
Profiles with structural impossibilities. Use the honeypot detection criteria from Section 7.3 to find these during your manual inspection.

### 4.3 The validation set JSON format

```json
[
  {
    "candidate_id": "CAND_0000001",
    "tier": 1,
    "note": "Backend/data engineer at consulting firm, no retrieval evidence, Canada-based"
  },
  {
    "candidate_id": "CAND_0000042",
    "tier": 3,
    "note": "6 years applied ML at product company, built FAISS-based search, good engagement"
  }
]
```

The `note` field is for your own reference during error analysis. It is not used by the code.

### 4.4 The metric calculation function

The harness lives in `train_weights.py` and is also callable standalone. It implements the exact formula from the submission spec:

```
Local Composite = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

NDCG implementation uses the standard formula with tier as the relevance score (tier 3 = 3, tier 2 = 2, tier 1 = 0, tier 0 = 0). MAP is computed across all tier-3 candidates as positives. Also compute and report P@5 as a diagnostic/tiebreak metric because the official submission tiebreaks use higher P@5 first, then higher P@10.

The harness function signature:

```python
def evaluate_ranking(ranked_candidate_ids: list[str], validation_set: list[dict]) -> dict:
    """
    ranked_candidate_ids: ordered list of all 100 candidate IDs from your submission, rank 1 first
    validation_set: loaded from metadata/validation_set.json
    Returns: dict with keys ndcg_10, ndcg_50, map, p_5, p_10, composite
    """
```

---

## 5. Phase 1 — Streaming Ingestion & Recall Filter

### 5.1 What this phase does

Reads `candidates.jsonl` or `candidates.jsonl.gz` line by line without loading the full ~465 MB into memory. For each candidate, checks high-recall profile and career text against a keyword index. If the candidate has zero matching tokens across all checked text, they are dropped from the rich-evidence pool but still retained in the full candidate feature store with low/default technical features. Everyone with at least one match goes to the staging file.

The purpose is not to find good candidates. It is to eliminate candidates who are categorically unrelated to technical roles — accountants, lawyers, civil engineers, teachers. This is a recall filter, not a precision filter. When in doubt, keep the candidate.

### 5.2 Fields to check

Check these fields after lowercasing:
- `profile.headline`
- `profile.current_title`
- `profile.summary`
- `profile.current_industry`
- `education[0].field_of_study` (use the first education entry only; if the array is empty, skip this field)
- every `career_history[i].title`
- every `career_history[i].description`

The filter is a recall filter, not a precision filter. It is better to keep too many candidates than to drop a plain-language fit whose career history mentions search, ranking, recommendation, or ML without a fashionable skill keyword.

### 5.3 The keyword index

```python
RECALL_TOKENS = {
    "ml", "ai", "machine", "learning", "data", "software", "backend",
    "developer", "engineer", "computer", "science", "nlp", "search",
    "retrieval", "ranking", "recommendation", "analytics", "research",
    "python", "deep", "neural", "model", "algorithm", "systems"
}
```

The tokenization is whitespace and punctuation split followed by lowercasing. Do not use stemming or fuzzy matching here — this is a speed filter, not a semantic filter.

### 5.4 Expected output size and validation

Expected output: 40,000 to 50,000 candidates in `artifacts/high_signal_pool.jsonl`.

If the output is below 30,000, the keyword index is too aggressive. Add broader tokens.  
If the output is above 60,000, that is acceptable — the tri-vector funnel will narrow it.  
Never tune this filter to go below 30,000. False negatives at this stage are unrecoverable.

### 5.5 Memory management

Stream line by line. Do not build a list of all candidates in memory. Write each passing candidate directly to the output file. The JSONL format allows this — one JSON object per line, no surrounding array brackets.

```python
import gzip
import json

def open_jsonl(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "rt", encoding="utf-8")

with open_jsonl(input_path) as f_in, open("artifacts/high_signal_pool.jsonl", "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        candidate = json.loads(line)
        if passes_recall_filter(candidate):
            f_out.write(line + "\n")
```

---

## 6. Phase 2 — Tri-Vector Retrieval Funnel

### 6.1 What this phase does

Takes the 40,000-50,000 staging candidates and narrows them to approximately 2,500-4,000 using three independent retrievers running in parallel. The union of all three retrievers' outputs becomes the rich-evidence pool for expensive semantic/evidence work. The final Parquet still contains all 100,000 candidate IDs with baseline/default features for candidates outside this pool.

The three-retriever design is not arbitrary. Each retriever catches a different type of genuine candidate:
- Retriever A (career embedding cosine similarity) catches candidates whose career narrative is semantically close to the JD even if they don't use the exact required keywords
- Retriever B (high-trust skill density) catches candidates who have strong, well-corroborated skill profiles
- Retriever C (keyword-matched evidence terms) catches candidates who explicitly use the technical vocabulary from the JD in their career descriptions

A candidate who is plain-spoken about their work may score low on Retriever A but still get captured by Retriever C if they used any of the target terms in their job descriptions.

### 6.2 The embedding model

Use `all-MiniLM-L6-v2` from the `sentence-transformers` library. This model produces 384-dimensional vectors. It runs on CPU in acceptable time for this dataset size. Do not use a larger model — the offline phase has no strict time limit but the embedding computation for 50,000 candidates still needs to complete in a reasonable timeframe.

Load the model once and reuse it for all embedding operations:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
```

### 6.3 The JD target vector

Build the JD query text by concatenating the most signal-rich sections of the job description into a single string. Use this exact text (do not modify it at runtime; hardcode it in the module):

```python
JD_QUERY_TEXT = """
Senior AI Engineer production embeddings retrieval systems sentence-transformers BGE E5 
vector databases Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS 
hybrid search semantic search dense retrieval Python evaluation frameworks NDCG MRR MAP 
A/B testing ranking systems recommendation systems learning to rank XGBoost neural reranking 
product company applied ML deployment real users inference optimization latency
"""
```

Embed this once at the start of the funnel phase:

```python
jd_vector = model.encode(JD_QUERY_TEXT, normalize_embeddings=True)
```

### 6.4 Retriever A — Career Vector Cosine Similarity

For each candidate in the staging pool, build the career text as:

```python
def build_career_text(candidate):
    profile = candidate["profile"]
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_industry", ""),
    ]
    for role in candidate["career_history"]:
        parts.append(role["title"])
        parts.append(role["description"])
    return " ".join(parts)
```

Encode all career texts in batches of 256 using `model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)`.

Compute cosine similarity against `jd_vector` using NumPy dot product (since vectors are already normalized, dot product equals cosine similarity). Store the resulting scalar `career_semantic_score` in the Parquet feature store. Do not require `rank.py` to load `sentence-transformers` or encode the JD at runtime.

Take the top 1,000 candidates by cosine similarity score. Save their candidate_ids to a set.

### 6.5 Retriever B — High-Trust Skill Density

This retriever does not use embeddings. For each candidate, compute a preliminary trust-weighted skill score:

```python
def quick_skill_density(candidate, target_skills):
    """
    target_skills: set of lowercase skill name substrings that indicate relevance
    Returns: float score representing density of trusted relevant skills
    """
    target_skills = {
        "embedding", "vector", "retrieval", "faiss", "pinecone", "qdrant",
        "milvus", "weaviate", "opensearch", "elasticsearch", "nlp", "bert",
        "transformer", "python", "ranking", "recommendation", "search",
        "ndcg", "mrr", "xgboost", "learning to rank", "sentence-transformer",
        "semantic", "dense", "sparse", "hybrid", "rerank", "llm", "fine-tun"
    }
    score = 0.0
    for skill in candidate["skills"]:
        name_lower = skill["name"].lower()
        if any(t in name_lower for t in target_skills):
            endorsements = skill.get("endorsements", 0)
            duration = skill.get("duration_months", 0)
            trust = (
                0.40 * min(endorsements / 20.0, 1.0) +
                0.40 * min(duration / 24.0, 1.0)
            )
            score += trust
    return score
```

This is a fast CPU operation. Sort all staging candidates by this score and take the top 1,000. Save their candidate_ids to a set.

### 6.6 Retriever C — Keyword-Matched Evidence Terms

This retriever searches the raw text of career history descriptions for specific technical terms. The intent is to catch candidates who explicitly name the technologies or methodologies required by the JD in their work descriptions, even if their job titles or skills sections are not keyword-dense.

```python
EVIDENCE_TERMS = {
    "faiss", "pinecone", "qdrant", "milvus", "weaviate", "opensearch",
    "elasticsearch", "dense retrieval", "sparse retrieval", "hybrid search",
    "semantic search", "vector search", "embedding", "sentence-transformer",
    "ndcg", "mrr", "mean average precision", "learning to rank", "xgboost",
    "rerank", "bm25", "inverted index", "ann search", "approximate nearest",
    "a/b test", "online eval", "offline eval", "ranking pipeline",
    "recommendation system", "retrieval system", "search ranking"
}

def evidence_term_count(candidate):
    text = " ".join(
        role["description"].lower() for role in candidate["career_history"]
    )
    return sum(1 for term in EVIDENCE_TERMS if term in text)
```

Sort all staging candidates by evidence term count and take the top 1,000 with count > 0. Save their candidate_ids to a set.

### 6.7 The union and operational pool

```python
operational_ids = retriever_a_ids | retriever_b_ids | retriever_c_ids
```

This union will typically contain 2,000 to 4,000 unique candidates. Load all of them from the staging file into memory — they now become the rich-evidence operational pool for the expensive parts of the feature factory.

If the operational pool exceeds 4,000 candidates, the evidence extraction phase may become noisy and slow. In that case, score each candidate using the quick_skill_density function and keep only the top 3,000 by that score.

**Recall verification — do this before running the feature factory:**

After building the operational pool, manually inspect 10-15 candidates from your validation set's Tier 3 group and confirm they are all present in the pool. If any Tier 3 candidate is missing, trace which retriever should have caught them and why it didn't. A missing Tier 3 candidate at this stage is unrecoverable — they will never appear in the final top 100. Fix the retriever thresholds before proceeding. This check takes 10 minutes and prevents the most damaging silent failure in the entire pipeline.

### 6.8 Full Feature Store Requirement

The operational pool is only the expensive-evidence pool. The runtime feature store must still contain one row for every candidate in the input dataset. For candidates outside `operational_candidate_ids.json`, compute cheap schema/behavioral fields normally and set technical fields to conservative defaults:

```python
career_semantic_score = 0.0
retrieval_count = 0
ranking_count = 0
evaluation_count = 0
production_count = 0
retrieval_snippet = ""
ranking_snippet = ""
evaluation_snippet = ""
production_snippet = ""
skill_trust_density = min(skill_trust_density, 0.05)
```

This keeps `rank.py` simple: it filters the full Parquet to the supplied input IDs, scores rows, and never needs to read raw candidate JSON for the full 100K evaluation.

---

## 7. Phase 3 — Offline Feature Factory

This phase creates the runtime feature store. It processes every candidate at least once for cheap scalar/schema features and applies expensive semantic/evidence extraction only to the rich-evidence pool (approximately 2,500-4,000 candidates). This is the computationally expensive phase. It has no time limit.

### 7.1 Component A — Skill Trust Scorer

For every skill in a candidate's skills array, compute a trust score. The trust score measures whether the skill is genuinely held or is a keyword-stuffer artifact.

**The formula:**

```
Skill Trust Score = 
    0.40 × min(endorsements / 20, 1.0) +
    0.40 × min(duration_months / 24, 1.0) +
    0.20 × CorroborationFlag
```

**Parameter definitions:**

`endorsements` — taken directly from `skill.endorsements`. If the field is missing, treat as 0.

`duration_months` — taken directly from `skill.duration_months`. If the field is missing, treat as 0. If it is present but 0, treat as 0.

`CorroborationFlag` — 1.0 if the skill name (lowercased, stripped of special characters) appears as a substring in any of the candidate's career_history description strings (lowercased). Otherwise 0.0.

Implementation note for CorroborationFlag: Use substring matching, not exact word matching. "FAISS" should match in "built a FAISS-based index". "Python" should match in "Python scripting and automation". Build the full career description text once per candidate before iterating over skills.

**The aggregate output per candidate:**

After computing individual skill trust scores, compute two aggregate values:

`skill_trust_density` — the sum of trust scores for skills whose name matches any token in the JD target skills list (same set used in Retriever B), divided by the number of JD target skills. This is the normalized density of trusted, JD-relevant skills.

`raw_skill_trust_sum` — the sum of all individual skill trust scores regardless of JD relevance. This measures overall profile credibility.

### 7.2 Component B — Profile Consistency Scorer (Honeypot Hard Gate)

This component detects structural anomalies that indicate a honeypot or fabricated profile.

**Hard Gate, Not Soft Penalty:**

Do not use a soft Consistency Score alone for honeypots:
- A single timeline violation drops score to 0.65 — NOT suppressed. Honeypot leaks into top 100.
- Both timeline + assessment violations drop score to 0.30 — NOT suppressed (condition was strictly <0.30). Still leaks.

The correct design: ANY structural impossibility = immediate hard exclusion (final_score = 0.0).

**Honeypot types identified in the submission spec:**

> *"~80 honeypots with subtly impossible profiles (e.g., 8 years of experience at a company founded 3 years ago; 'expert' proficiency in 10 skills with 0 years used)"*

This means the hard gate must catch THREE types:
1. **Skill duration > total YoE** (classic honeypot: Python for 80 months when YoE = 3)
2. **Expert/advanced skill with 0 months duration** (spec explicitly says "0 years used")
3. **Per-role tenure > total YoE** (company founded 3 years ago but candidate claims 8 years there)

**Step 1: Hard Honeypot Gate (run first, before ANY scoring)**

```python
def is_honeypot(candidate) -> bool:
    """
    Returns True if the candidate has ANY structural impossibility.
    These are forced to relevance tier 0 in the ground truth.
    A True return means final_score = 0.0 — excluded from top 100.
    """
    yoe = candidate["profile"].get("years_of_experience", 0)
    yoe_months = yoe * 12
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    # Check 1: Skill duration > total career length (+6 month grace)
    for skill in skills:
        duration = skill.get("duration_months", 0)
        if duration > yoe_months + 6:  # e.g., Python for 80mo when YoE=3
            return True

    # Check 2: Expert/advanced skill with ZERO duration ("0 years used")
    # Spec says this is an explicit honeypot pattern.
    for skill in skills:
        if skill.get("proficiency") in ("expert", "advanced"):
            duration = skill.get("duration_months", None)
            if duration is not None and duration == 0:
                return True  # Claims expert with 0 months of use — impossible

    # Check 3: Assessment contradiction — expert/advanced skill with low assessment score
    for skill in skills:
        if skill.get("proficiency") in ("expert", "advanced"):
            skill_name = skill["name"]
            score = assessment_scores.get(skill_name)
            if score is not None and score < 40:
                return True

    # Check 4: Per-role tenure > total YoE ("8 years at company founded 3 years ago")
    # If any single career role's duration_months > yoe_months + 12 grace, it's impossible.
    for role in career:
        role_duration = role.get("duration_months", 0)
        if role_duration > yoe_months + 12:  # +12 month grace for rounding
            return True

    return False
```

**Step 2: Secondary Consistency Score (for non-honeypot profiles)**

For candidates who passed the hard gate, compute a soft consistency score to distinguish
slightly inconsistent profiles from clean ones. This is NOT a honeypot gate — it is a
mild seniority sanity check used as one of the 7 scoring components.

```
Consistency Score = 1.0 - clip(0.30 × N_titles, 0.0, 1.0)
```

`N_titles` — binary counter, increments by 1 if `years_of_experience > 6` AND `current_title` contains any of these substrings (case-insensitive): "junior", "associate", "intern", "trainee", "entry". This catches profiles where someone claims 8 years of experience but holds a junior title — a structural inconsistency.

For candidates who passed the hard gate, Consistency Score will be 0.70 (if title mismatch) or 1.0 (clean profile). This is used as S_Integrity in the scoring formula.

Store the boolean result as `is_honeypot` in the Parquet feature store. `rank.py` must never recompute honeypots from raw JSON during full evaluation; it only reads the precomputed flag and forces `final_score = 0.0` before selecting the top 100.

### 7.3 Component C — Product Company Classifier

This is a deterministic rule-based classifier, not an ML model. For each role in `career_history`, classify the company as either a product company or a services/consulting company.

**Definite consulting/services companies** (hard-coded list, case-insensitive match):

```python
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "ltimindtree", "niit technologies", "zensar", "mastech",
    "syntel", "kpit", "cyient", "birlasoft", "infotech enterprises",
    "persistent systems" 
}
```

For each career role, check if `company.lower()` contains any token from the consulting firms list. If yes, classify as consulting. Also check `industry` — if it equals "IT Services" or "Consulting" or "Outsourcing", classify as consulting regardless of company name.

Note: `current_company_size` is not sufficient to determine product vs consulting. Large companies can be either.

**Computing the product ratio:**

```python
def compute_product_ratio(candidate):
    total_months = 0
    product_months = 0
    for role in candidate["career_history"]:
        duration = role.get("duration_months", 0)
        total_months += duration
        if not is_consulting_role(role):
            product_months += duration
    if total_months == 0:
        return 0.5  # unknown, neutral

### 7.3 Component C — Product Company Classifier

This is a deterministic rule-based classifier, not an ML model. For each role in `career_history`, classify the company as either a product company or a services/consulting company.

**Definite consulting/services companies** (hard-coded list, case-insensitive match):

```python
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "ltimindtree", "niit technologies", "zensar", "mastech",
    "syntel", "kpit", "cyient", "birlasoft", "infotech enterprises",
    "persistent systems" 
}
```

For each career role, check if `company.lower()` contains any token from the consulting firms list. If yes, classify as consulting. Also check `industry` — if it equals "IT Services" or "Consulting" or "Outsourcing", classify as consulting regardless of company name.

Note: `current_company_size` is not sufficient to determine product vs consulting. Large companies can be either.

**Computing the product ratio:**

```python
def compute_product_ratio(candidate):
    total_months = 0
    product_months = 0
    for role in candidate["career_history"]:
        duration = role.get("duration_months", 0)
        total_months += duration
        if not is_consulting_role(role):
            product_months += duration
    if total_months == 0:
        return 0.5  # unknown, neutral
    return product_months / total_months
```

The output is a float between 0.0 and 1.0. A candidate who has spent their entire career at product companies scores 1.0. A consulting-only candidate scores 0.0.

### 7.4 Component D — Regex-First Evidence Extractor

The default implementation is deterministic regex/pattern extraction over the operational pool. This is fast, explainable, easy to reproduce, and avoids adding a 4.5 GB local LLM artifact that competes with the official 5 GB disk/intermediate-state limit. Do not put any LLM model or LLM package on the runtime path.

```python
RETRIEVAL_PATTERNS = [
    r"faiss", r"pinecone", r"qdrant", r"milvus", r"weaviate",
    r"opensearch", r"elasticsearch", r"dense retrieval",
    r"vector search", r"embedding", r"semantic search",
    r"ann\b", r"approximate nearest", r"sentence.transformer",
    r"bi.encoder", r"cross.encoder", r"dense.encoder"
]

RANKING_PATTERNS = [
    r"learning.to.rank", r"xgboost.*rank", r"lambdamart",
    r"pairwise.*rank", r"listwise", r"ranking.*pipeline",
    r"relevance.*score", r"rerank", r"bm25"
]

MATCHING_PATTERNS = [
    r"recommendation.system", r"recsys", r"collaborative.filtering",
    r"content.based.filtering", r"matching.engine", r"candidate.matching",
    r"personalization.engine", r"match.score"
]

EVALUATION_PATTERNS = [
    r"ndcg", r"mrr\b", r"mean.average.precision", r"map\b.*eval",
    r"a/b.test", r"online.*eval", r"offline.*eval",
    r"precision.at", r"recall.at", r"f1.*rank"
]

PRODUCTION_PATTERNS = [
    r"produc.*deploy", r"latency", r"inference.*serv",
    r"real.user", r"live.*system", r"million.*request",
    r"billion.*query", r"serving.*infrastructure",
    r"qps\b", r"p99", r"p95"
]
```

For each pattern group, count the number of distinct pattern matches in the concatenated career/profile text. Store as `retrieval_count`, `ranking_count`, `matching_count`, `evaluation_count`, `production_count`.

**The output feature:**

The feature stored in the parquet is a single integer per evidence category:

```python
retrieval_count = regex_match_count(RETRIEVAL_PATTERNS, full_profile_text)
ranking_count = regex_match_count(RANKING_PATTERNS, full_profile_text)
matching_count = regex_match_count(MATCHING_PATTERNS, full_profile_text)
evaluation_count = regex_match_count(EVALUATION_PATTERNS, full_profile_text)
production_count = regex_match_count(PRODUCTION_PATTERNS, full_profile_text)
```

Also store the first snippet from each evidence category as individual string columns in the Parquet: `retrieval_snippet`, `ranking_snippet`, `matching_snippet`, `evaluation_snippet`, and `production_snippet`. The reasoning generator reads these scalar strings at runtime. Do not require runtime access to list-valued evidence arrays.

**Additional JD-derived boolean detectors:**

These are computed from lowercased `skills.name`, `profile.headline`, `profile.current_title`, and all `career_history.description` text.

```python
FRAMEWORK_DEMO_TERMS = [
    r"langchain", r"llamaindex", r"openai api", r"chatgpt api",
    r"prompt engineering", r"wrapper", r"tutorial", r"demo app",
    r"toy project", r"built a chatbot"
]

EXTERNAL_VALIDATION_TERMS = [
    r"open.source", r"github", r"published", r"publication", r"paper",
    r"conference", r"talk", r"speaker", r"blog", r"technical article",
    r"maintainer", r"contributor"
]

HRTECH_MARKETPLACE_TERMS = [
    r"hr.tech", r"recruit", r"talent", r"candidate matching",
    r"job matching", r"marketplace", r"two.sided marketplace",
    r"matching engine", r"recommendation marketplace"
]

DISTRIBUTED_INFERENCE_TERMS = [
    r"distributed system", r"microservice", r"high throughput",
    r"low latency", r"latency optimization", r"inference optimization",
    r"model serving", r"feature store", r"kafka", r"spark",
    r"autoscaling", r"qps", r"p95", r"p99"
]
```

Compute:

```python
systems_evidence_count = retrieval_count + ranking_count + matching_count + evaluation_count + production_count

is_framework_demo_only = (
    regex_any(FRAMEWORK_DEMO_TERMS, full_profile_text)
    and systems_evidence_count <= 1
    and product_ratio < 0.50
)

lacks_external_validation = (
    yoe >= 5
    and github_activity_score <= 0
    and not regex_any(EXTERNAL_VALIDATION_TERMS, full_profile_text)
)

has_hrtech_marketplace_exposure = regex_any(HRTECH_MARKETPLACE_TERMS, full_profile_text)
has_distributed_inference_exposure = regex_any(DISTRIBUTED_INFERENCE_TERMS, full_profile_text)
```

Do not use these detectors as hard exclusions. They are weak-to-moderate JD modifiers because profile text may omit public validation even when it exists.

### 7.5 Component E — Embedding Generation

Generate semantic scores offline using `all-MiniLM-L6-v2`.

**Career embedding:**

Build the career text string exactly as in Section 6.4 (current_title + headline + summary + current_industry + all role titles + all role descriptions concatenated). Truncate to 512 tokens if needed.

**Trusted skills text:**

Build a string containing only the names of skills whose individual Skill Trust Score (computed in Component A) is strictly greater than 0.5. Concatenate with spaces. This is useful for offline analysis and optional retrieval, but the default runtime path should use scalar `skill_trust_density` rather than loading skill embeddings.

```python
trusted_skills_text = " ".join(
    skill["name"] for skill in candidate["skills"]
    if compute_skill_trust_score(skill, career_text) > 0.5
)
```

**Serialization:**

After processing the rich-evidence pool, write scalar semantic scores into `candidate_features.parquet`. Saving full embedding arrays is optional and should be avoided in the final runtime package unless needed for analysis, because the official ranking step should stay small, deterministic, and free of model-loading/network-cache risk.

### 7.6 The Parquet Feature Store

After all components run, write `artifacts/candidate_features.parquet` with one row per candidate in the input dataset, not just the operational pool. The rich-evidence pool receives full semantic/evidence features. Low-signal candidates outside that pool receive explicit zero/default technical features, so `rank.py` can score the full 100,000 candidates without runtime feature extraction.

| Column | Type | Source | Notes |
|--------|------|--------|-------|
| candidate_id | string | profile | Primary key |
| yoe | float | profile.years_of_experience | |
| current_title | string | profile.current_title | |
| current_company | string | profile.current_company | |
| location | string | profile.location | |
| country | string | profile.country | |
| skill_trust_density | float | Component A | |
| raw_skill_trust_sum | float | Component A | |
| consistency_score | float | Component B | 0.70 or 1.0 (post hard-gate) |
| is_honeypot | bool | Component B hard gate | True = score forced to 0.0 |
| product_ratio | float | Component C | 0.0-1.0 |
| career_semantic_score | float | Component E | JD/career cosine similarity rescaled to 0-1; 0 for low-signal candidates |
| retrieval_count | int | Component D | |
| ranking_count | int | Component D | |
| matching_count | int | Component D | |
| evaluation_count | int | Component D | |
| production_count | int | Component D | |
| retrieval_snippet | string | Component D | First retrieval phrase or "" |
| ranking_snippet | string | Component D | First ranking phrase or "" |
| matching_snippet | string | Component D | First matching phrase or "" |
| evaluation_snippet | string | Component D | First eval phrase or "" |
| production_snippet | string | Component D | First deployment phrase or "" |
| notice_period_days | int | redrob_signals | |
| recruiter_response_rate | float | redrob_signals | |
| open_to_work_flag | bool | redrob_signals | |
| last_active_date | string | redrob_signals | |
| interview_completion_rate | float | redrob_signals | |
| github_activity_score | float | redrob_signals | -1 = no GitHub |
| offer_acceptance_rate | float | redrob_signals | -1 = no history |
| willing_to_relocate | bool | redrob_signals | |
| preferred_work_mode | string | redrob_signals | |
| avg_response_time_hours | float | redrob_signals | |
| applications_submitted_30d | int | redrob_signals | |
| profile_views_received_30d | int | redrob_signals | recruiter profile views in last 30d |
| search_appearance_30d | int | redrob_signals | recruiter search appearances in last 30d |
| saved_by_recruiters_30d | int | redrob_signals | |
| profile_completeness_score | float | redrob_signals | |
| endorsements_received | int | redrob_signals | total profile endorsements |
| verified_email | bool | redrob_signals | |
| verified_phone | bool | redrob_signals | |
| linkedin_connected | bool | redrob_signals | |
| avg_tenure_months | float | computed | total_months / num_roles |
| num_roles | int | computed | len(career_history) |
| avg_description_length | float | computed | average length of job description strings |
| is_title_chaser | bool | computed | avg_tenure<18 AND num_roles>=3 |
| is_pure_research | bool | computed | Only academic roles in history |
| is_architect_only | bool | computed | Latest=Architect/VP AND yoe>8 |
| is_framework_demo_only | bool | computed | LangChain/OpenAI demo/tutorial-heavy profile with weak systems evidence |
| lacks_external_validation | bool | computed | yoe>=5, no GitHub/public papers/talks/open-source signals |
| has_hrtech_marketplace_exposure | bool | computed | HR-tech/recruiting/marketplace/matching exposure in career text |
| has_distributed_inference_exposure | bool | computed | Distributed systems, serving, latency, inference optimization evidence |

**CRITICAL NOTE on evidence snippet columns:**

`retrieval_snippet`, `ranking_snippet`, `matching_snippet`, `evaluation_snippet`, `production_snippet` must be saved as individual string columns in the Parquet. The reasoning generator in `src/explainer.py` reads these directly at inference time. If empty string, the generator uses the fallback path — that is correct and expected. What is WRONG is having them absent from the Parquet entirely, which causes a KeyError crash in the reasoning generator.

---

## 8. Phase 4 — Weight Optimization Loop

### 8.1 What this does

Instead of hardcoding the 7 component weights, use the validation set to find the weight vector that maximizes the local composite NDCG score. This is an offline step that runs once after the feature factory completes and before rank.py is finalized.

### 8.2 The scoring formula to be optimized

The weights w1 through w7 correspond to:

```
w1 = weight for S_Alignment (Project Alignment Score)
w2 = weight for S_Trust (Skill Trust Density)
w3 = weight for S_Semantic (Semantic Similarity)
w4 = weight for S_Integrity (Profile Consistency)
w5 = weight for S_Engagement (Platform Engagement)
w6 = weight for S_Seniority (Seniority Match)
w7 = weight for S_Proximity (Geographic Proximity)
```

Starting values before optimization:

```python
x0 = [0.30, 0.25, 0.15, 0.10, 0.10, 0.05, 0.05]
```

The optimizer normalizes weights at each step so they sum to 1.0.

### 8.3 The optimization loop

```python
import scipy.optimize as opt
import numpy as np

def objective_loss(weights):
    normalized_w = np.array(weights) / np.sum(weights)
    scores = compute_all_scores(validation_features, normalized_w)
    ranked_ids = rank_by_scores(scores)
    composite = evaluate_ranking(ranked_ids, validation_set)["composite"]
    return 1.0 - composite

result = opt.minimize(
    objective_loss,
    x0=[0.30, 0.25, 0.15, 0.10, 0.10, 0.05, 0.05],
    method="Nelder-Mead",
    options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4}
)

optimized_weights = result.x / np.sum(result.x)
```

Save optimized weights:

```python
import json
with open("artifacts/optimized_weights.json", "w") as f:
    json.dump({
        "w1_alignment": float(optimized_weights[0]),
        "w2_trust": float(optimized_weights[1]),
        "w3_semantic": float(optimized_weights[2]),
        "w4_integrity": float(optimized_weights[3]),
        "w5_engagement": float(optimized_weights[4]),
        "w6_seniority": float(optimized_weights[5]),
        "w7_proximity": float(optimized_weights[6])
    }, f, indent=2)
```

### 8.4 Limitations of this optimization

The validation set is hand-labeled by you. Your labels are your interpretation of what Redrob considers a strong candidate. They may not perfectly match Redrob's hidden ground truth. This means the optimized weights are the best weights given your labeling judgment, not the objectively best weights.

To reduce this risk: when labeling, follow the JD criteria strictly. Err on the side of being conservative with Tier 3 labels — only label a candidate Tier 3 if you are confident they would pass a Redrob recruiter's first screen. If in doubt, label Tier 2.

**Overfitting guard:** After optimization, check the optimized weight vector manually. If any single weight has converged above 0.60, treat this as a sign that the validation set is too small or too homogeneous. In that case, add more Tier 2 candidates to the validation set (these borderline cases are what force the optimizer to find real discriminating weights) and re-run. Do not submit with a weight vector dominated by a single component.

---

## 9. Phase 5 — Runtime Scoring Engine (rank.py)

This is the script that runs at evaluation time. It must complete in under 5 minutes on CPU with 16 GB RAM, <=5 GB disk/intermediate state, and no network access.

### 9.1 Startup and loading

```python
import argparse
import gzip
import json
import pandas as pd
import numpy as np
from src.explainer import generate_dynamic_reasoning
from src.fast_extractor import fast_extract_features  # on-the-fly extractor for sandbox

def load_input_candidates(path: str) -> list[dict]:
    """Load candidates from .jsonl or .jsonl.gz input file."""
    open_fn = gzip.open if path.endswith(".gz") else open
    candidates = []
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Path to .jsonl or .jsonl.gz input file")
    parser.add_argument("--out", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    # Step 1: Load the candidate input file passed by the evaluator.
    # This may be candidates.jsonl (full 100K) OR a small sample (sandbox <=100).
    # rank.py MUST only output candidate_ids that exist in this file.
    input_candidates = load_input_candidates(args.candidates)
    input_ids = {c["candidate_id"] for c in input_candidates}

    # Step 2: Load precomputed scalar artifacts
    features = pd.read_parquet("artifacts/candidate_features.parquet")
    weights = json.load(open("artifacts/optimized_weights.json"))
    run_metadata = json.load(open("artifacts/run_metadata.json"))

    # Step 3: Filter parquet to ONLY the candidates present in the input file.
    # For the full 100K evaluation: almost all will be in the parquet.
    # For sandbox (small sample): may have few or zero parquet hits.
    features_in_input = features[features["candidate_id"].isin(input_ids)].copy()
    precomputed_ids = set(features_in_input["candidate_id"].tolist())

    # Step 4: On-the-fly fast extractor only for candidates NOT in precomputed parquet.
    # In full evaluation this should be zero because the parquet contains all 100K IDs.
    # This exists for sandbox/demo samples that contain custom or subset candidate IDs.
    missing_candidates = [c for c in input_candidates if c["candidate_id"] not in precomputed_ids]
    if missing_candidates:
        # fast_extract_features uses the same regex-based feature extraction as the
        # offline factory but runs inline. Returns a DataFrame with the same schema.
        missing_features = fast_extract_features(missing_candidates)
        features_in_input = pd.concat([features_in_input, missing_features], ignore_index=True)

    # All subsequent scoring operates on features_in_input only.
    features = features_in_input.reset_index(drop=True)
```

**CRITICAL: Why this matters for the sandbox requirement.**

The submission spec Section 10.5 requires: *"Accept a small candidate sample (≤100 candidates) as input... run your ranking system end-to-end."* If rank.py only loads from the precomputed parquet and ignores the `--candidates` argument, sandbox evaluation fails immediately — the output will either be empty or contain candidate IDs that don't exist in the input file, violating Rule F5.

### 9.2 The 7 scoring components

Compute each component as a NumPy array over all candidates in the features dataframe. All computations are vectorized — no Python loops over candidates.

**S_Alignment — Project Alignment Score**

```python
# The JD specifically asks for candidates who have worked across specific domains:
# Retrieval, Ranking, Matching/Recommendation, and Evaluation.
# We reward candidates exponentially more if they have cross-domain evidence.
domain_coverage = (
    (features["retrieval_count"] > 0).astype(int) + 
    (features["ranking_count"] > 0).astype(int) + 
    (features["matching_count"] > 0).astype(int) + 
    (features["evaluation_count"] > 0).astype(int)
)

evidence_sum = (
    features["retrieval_count"] + 
    features["ranking_count"] + 
    features["matching_count"] +
    features["evaluation_count"] + 
    features["production_count"]
)

# A candidate with evidence in multiple domains gets a massive multiplier.
# This explicitly shifts the score away from generic AI engineers toward JD-aligned domain experts.
S_Alignment = (1.0 + 0.5 * domain_coverage.values) * np.log1p(evidence_sum.values)
alignment_max = S_Alignment.max()
S_Alignment = S_Alignment / alignment_max if alignment_max > 0 else np.zeros_like(S_Alignment)
```

**S_Trust — Skill Trust Density**

```python
S_Trust = features["skill_trust_density"].values.clip(0, 1)
```

**S_Semantic — Career Embedding Cosine Similarity**

```python
# Precomputed offline from career/profile text and the JD query vector.
# Stored as a scalar to avoid loading sentence-transformers during rank.py.
S_Semantic = features["career_semantic_score"].fillna(0.0).values.clip(0, 1)
```

**S_Integrity — Profile Consistency Score**

```python
S_Integrity = features["consistency_score"].values.clip(0, 1)
```

**S_Engagement — Platform Engagement (Expanded to 6 signals)**

```python
# RecencyScore: days since last_active_date
from datetime import date

# Deterministic reference date: use the max last_active_date observed during preprocessing,
# stored in artifacts/run_metadata.json. Do not use date.today(), or reproduction scores
# will drift depending on when the organizers run rank.py.
reference_date = date.fromisoformat(run_metadata["reference_date"])

def compute_recency(last_active_str):
    if pd.isna(last_active_str):
        return 0.1
    days_inactive = (reference_date - date.fromisoformat(last_active_str)).days
    if days_inactive <= 30:
        return 1.0
    elif days_inactive <= 60:
        return 0.8
    elif days_inactive <= 90:
        return 0.6
    elif days_inactive <= 180:
        return 0.4
    else:
        return 0.1  # >180 days: practically unreachable per JD guidance

recency_scores = features["last_active_date"].apply(compute_recency).values
response_rates = features["recruiter_response_rate"].fillna(0.5).values
open_to_work = features["open_to_work_flag"].astype(float).values

# interview_completion_rate: reliability once engaged. <0.50 = ghost risk.
interview_completion = features["interview_completion_rate"].fillna(0.5).values

# applications_submitted_30d: active job seeker signal (>0 = actively looking now)
apps_submitted = features["applications_submitted_30d"].fillna(0).clip(upper=5).values / 5.0

# avg_response_time_hours: slow responders (<72h acceptable, >168h = mild penalty)
def response_time_score(hours):
    if pd.isna(hours) or hours <= 0:
        return 0.5  # unknown, neutral
    elif hours <= 24:
        return 1.0
    elif hours <= 72:
        return 0.8
    elif hours <= 168:
        return 0.5
    else:
        return 0.2

response_time_scores = features["avg_response_time_hours"].apply(response_time_score).values

S_Engagement = (
    0.30 * recency_scores +          # Most important: are they even on the platform?
    0.30 * response_rates +           # Most important: will they respond?
    0.15 * open_to_work +             # Explicit availability flag
    0.15 * interview_completion +     # Reliability once engaged
    0.05 * apps_submitted +           # Active job seeker bonus
    0.05 * response_time_scores       # Practical reachability
)
```

**Secondary Behavioral Modifiers — Visibility, Legitimacy, and Reliability**

These use the remaining build-relevant Redrob signals as light multiplicative modifiers. They should never dominate technical fit.

```python
# Recruiter market validation. Use log compression so high counts do not swamp fit.
views = features["profile_views_received_30d"].fillna(0).clip(lower=0).values
search_appearances = features["search_appearance_30d"].fillna(0).clip(lower=0).values
saved = features["saved_by_recruiters_30d"].fillna(0).clip(lower=0).values
endorsements = features["endorsements_received"].fillna(0).clip(lower=0).values

market_validation_score = np.clip(
    0.25 * np.log1p(views) / np.log1p(100) +
    0.20 * np.log1p(search_appearances) / np.log1p(100) +
    0.35 * np.log1p(saved) / np.log1p(20) +
    0.20 * np.log1p(endorsements) / np.log1p(100),
    0.0,
    1.0
)
market_validation_modifier = 1.0 + 0.05 * market_validation_score

# Basic legitimacy. Missing LinkedIn is mild; missing verified email/phone is more concerning.
verified_email = features["verified_email"].fillna(False).astype(bool).values
verified_phone = features["verified_phone"].fillna(False).astype(bool).values
linkedin_connected = features["linkedin_connected"].fillna(False).astype(bool).values

legitimacy_score = (
    0.40 * verified_email.astype(float) +
    0.40 * verified_phone.astype(float) +
    0.20 * linkedin_connected.astype(float)
)
legitimacy_modifier = 0.95 + 0.05 * legitimacy_score

# Profile completeness and offer acceptance are light reliability modifiers.
profile_completeness = features["profile_completeness_score"].fillna(70).clip(0, 100).values
profile_completeness_modifier = np.where(
    profile_completeness < 50,
    0.90,
    1.0
)

offer_acceptance = features["offer_acceptance_rate"].fillna(-1).values
offer_acceptance_modifier = np.where(
    offer_acceptance < 0,
    1.0,  # no history: neutral
    np.where(offer_acceptance < 0.30, 0.90, 1.0)
)

# Work-mode fit for a hybrid Pune/Noida role. Keep mild because the JD says cadence is flexible.
preferred_work_mode = features["preferred_work_mode"].fillna("flexible").str.lower().values
work_mode_modifier = np.where(
    preferred_work_mode == "remote",
    0.95,
    1.0
)
```

**S_Seniority — Experience Alignment (Asymmetric Window)**

```python
yoe = features["yoe"].values
# The JD says 5-9 years is a RANGE, NOT A REQUIREMENT.
# "Some people hit senior judgment at 4 years; some never hit it after 15."
# Use an asymmetric window: full score for 5-10 years, soft taper below/above.
# A 4-year candidate gets 0.60 (not crushed) if other signals are strong.
# A 12-year candidate gets 0.75 (slightly penalized for over-seniority).

def compute_seniority(yoe_val):
    if 5.0 <= yoe_val <= 9.0:
        return 1.0  # Sweet spot per JD
    elif 4.0 <= yoe_val < 5.0:
        return 0.60 + 0.40 * (yoe_val - 4.0)  # Linear taper from 0.60 to 1.0
    elif 9.0 < yoe_val <= 12.0:
        return 1.0 - 0.10 * (yoe_val - 9.0)   # Gentle taper from 1.0 to 0.70
    elif yoe_val < 4.0:
        return max(0.20, 0.60 - 0.10 * (4.0 - yoe_val))  # Steep drop below 4
    else:  # >12 years
        return max(0.50, 0.70 - 0.05 * (yoe_val - 12.0))  # Mild drop above 12

S_Seniority = np.vectorize(compute_seniority)(yoe)
```

**S_Proximity — Geographic Alignment (Fixed: INDIA_ADJACENT now used)**

```python
# Tier 1 (Best): Pune, Noida — company offices are here
# Tier 2 (Welcome per JD explicitly): Hyderabad, Mumbai, Delhi NCR
# Tier 3 (Tier-1 Indian cities, willing to relocate): Bangalore, Chennai, Kolkata, etc.
# Tier 4 (India, willing): anywhere else in India, willing to relocate
# Tier 5 (India, not willing): anywhere in India, not relocating
# Tier 6 (Outside India): case-by-case, no visa sponsorship

PUNE_NOIDA_CITIES = {
    "pune", "noida", "greater noida", "delhi", "new delhi", "gurugram",
    "gurgaon", "faridabad", "ghaziabad"
}

# JD explicitly says: "Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply"
JD_WELCOME_CITIES = {"hyderabad", "mumbai", "delhi", "pune"}

# Other Tier-1 Indian cities with large tech talent pools
INDIA_ADJACENT = {
    "bangalore", "bengaluru", "chennai", "kolkata",
    "ahmedabad", "indore", "jaipur", "chandigarh", "kochi"
}

def compute_proximity(row):
    location = str(row["location"]).lower()
    country = str(row["country"]).lower()
    willing = bool(row["willing_to_relocate"])

    # Tier 1: In Pune or Noida area (exact office cities)
    if any(city in location for city in PUNE_NOIDA_CITIES):
        return 1.0

    # Tier 2: JD-explicitly-welcomed cities (Hyderabad, Mumbai, Delhi NCR)
    if any(city in location for city in JD_WELCOME_CITIES):
        return 0.95 if willing else 0.90

    # Tier 3: Other major Indian tech hubs, willing to relocate
    if any(city in location for city in INDIA_ADJACENT):
        if willing:
            return 0.85  # Explicitly: Tier-1 Indian city + willing = good fit
        else:
            return 0.70  # Adjacent Indian city but not willing

    # Tier 4: India (unknown city), willing to relocate
    if country == "india" and willing:
        return 0.75

    # Tier 5: India, not willing
    if country == "india":
        return 0.60

    # Tier 6: Outside India, willing — case-by-case per JD, no visa sponsorship
    if willing:
        return 0.40

    # Outside India, not willing
    return 0.20

S_Proximity = features.apply(compute_proximity, axis=1).values
```

### 9.3 The weighted combination

```python
w = np.array([
    weights["w1_alignment"],
    weights["w2_trust"],
    weights["w3_semantic"],
    weights["w4_integrity"],
    weights["w5_engagement"],
    weights["w6_seniority"],
    weights["w7_proximity"]
])

# --- SIGNAL CLIPPING: Clip all behavioral signals to their documented valid ranges.
# Prevents corrupted or out-of-range values from inflating scores silently.
# Per redrob_signals_doc: response_rate 0.0-1.0, github_score -1 to 100,
# interview_completion 0.0-1.0, offer_acceptance -1 to 1.0
features["recruiter_response_rate"] = features["recruiter_response_rate"].clip(0.0, 1.0)
features["interview_completion_rate"] = features["interview_completion_rate"].clip(0.0, 1.0)
features["github_activity_score"] = features["github_activity_score"].clip(-1, 100)
features["profile_completeness_score"] = features["profile_completeness_score"].clip(0, 100)
features["offer_acceptance_rate"] = features["offer_acceptance_rate"].clip(-1.0, 1.0)
features["notice_period_days"] = features["notice_period_days"].clip(0, 180)

raw_scores = (
    w[0] * S_Alignment +
    w[1] * S_Trust +
    w[2] * S_Semantic +
    w[3] * S_Integrity +
    w[4] * S_Engagement +
    w[5] * S_Seniority +
    w[6] * S_Proximity
)
```

### 9.4 Soft modifiers

Apply after the base score. These are multipliers, not additive components.

**Product Exposure Ratio:**

```python
product_ratio = features["product_ratio"].values
product_modifier = 0.50 + 0.50 * product_ratio
# A consulting-only candidate (ratio=0.0) gets multiplier 0.50
# A full product-company candidate (ratio=1.0) gets multiplier 1.00
# NOTE: per JD, if candidate is CURRENTLY at consulting but has prior product exp,
# product_ratio will already reflect this correctly since it's a time-weighted average.
```

**Notice Period Gradient:**

```python
def notice_modifier(days):
    if days <= 30:
        return 1.00  # JD: ideal, can buy out up to 30 days
    elif days <= 60:
        return 0.85
    elif days <= 90:
        return 0.65  # JD: bar gets higher
    else:
        return 0.30  # JD: significant concern

notice_days = features["notice_period_days"].fillna(60).values
notice_modifiers = np.vectorize(notice_modifier)(notice_days)
```

**Honeypot Hard Exclusion (replaces old soft suppression):**

```python
# is_honeypot was computed in preprocessing and stored in the parquet.
# ANY honeypot = final score of 0.0, excluded from top 100 entirely.
honeypot_mask = features["is_honeypot"].values.astype(bool)
```

**Title-Chaser Penalty (JD explicit disqualifier):**

```python
# The JD says: "switching every 1.5 years for title progression — we're not a fit."
title_chaser_mask = features["is_title_chaser"].values.astype(bool)
title_chaser_modifier = np.where(title_chaser_mask, 0.85, 1.0)
```

**GitHub Activity Bonus (AI engineer role, active GitHub is a positive signal):**

```python
github_scores = features["github_activity_score"].fillna(-1).values
# -1 = no GitHub linked → neutral (multiplier = 1.0)
# 0-100 = GitHub activity score → bonus up to +10%
github_modifier = np.where(
    github_scores > 0,
    1.0 + 0.10 * (github_scores / 100.0),  # Max +10% bonus at score=100
    1.0  # -1 or 0 = no bonus, no penalty
)
```

**Research Background Penalty (JD explicit disqualifier):**

```python
# Candidates who spent their entire career in research roles (no production deployment)
# Per JD: "spent career in pure research environments without production deployment — we will not move forward."
research_mask = features["is_pure_research"].values.astype(bool)
research_modifier = np.where(research_mask, 0.50, 1.0)
```

**Architect/Code-Stopped Penalty (JD explicit disqualifier):**

```python
# "Senior engineer who hasn't written production code in 18 months — probably not move forward."
architect_mask = features["is_architect_only"].values.astype(bool)
architect_modifier = np.where(architect_mask, 0.85, 1.0)
```

**Framework-Demo Penalty (JD explicit anti-signal):**

```python
# JD rejects profiles whose AI experience is mostly recent framework wrappers,
# tutorials, or demos rather than production retrieval/ranking systems.
framework_demo_mask = features["is_framework_demo_only"].values.astype(bool)
framework_demo_modifier = np.where(framework_demo_mask, 0.80, 1.0)
```

**External Validation Penalty (JD explicit anti-signal):**

```python
# Closed-source-only proprietary work for 5+ years without GitHub, papers,
# talks, OSS, or other external validation is a mild negative signal.
external_validation_mask = features["lacks_external_validation"].values.astype(bool)
external_validation_modifier = np.where(external_validation_mask, 0.90, 1.0)
```

**Domain and Scale Nice-to-Have Bonuses:**

```python
hrtech_marketplace_bonus = np.where(
    features["has_hrtech_marketplace_exposure"].values.astype(bool),
    1.03,
    1.0
)

distributed_inference_bonus = np.where(
    features["has_distributed_inference_exposure"].values.astype(bool),
    1.04,
    1.0
)
```

**Written Communication / Profile Detail Score:**
*The JD says: "We work async-first and write a lot. If you find writing painful, you'll find this role painful." A candidate who leaves job descriptions blank or extremely short is a poor fit for this culture. We calculate the average length of their career description strings and apply a penalty for minimal writing.*

```python
avg_desc_len = features["avg_description_length"].fillna(0).values
writing_detail_modifier = np.where(
    avg_desc_len >= 250.0,
    1.00,  # Detailed, clear communicator
    np.where(
        avg_desc_len >= 100.0,
        0.98,  # Decent writing detail
        0.93   # Short/minimal descriptions; writing is likely painful
    )
)
```

**Combined final score:**

```python
raw_scores = (
    w[0] * S_Alignment +
    w[1] * S_Trust +
    w[2] * S_Semantic +
    w[3] * S_Integrity +
    w[4] * S_Engagement +
    w[5] * S_Seniority +
    w[6] * S_Proximity
)

# Apply all soft multipliers
final_scores = (
    raw_scores
    * product_modifier
    * notice_modifiers
    * title_chaser_modifier
    * github_modifier
    * research_modifier
    * architect_modifier
    * framework_demo_modifier
    * external_validation_modifier
    * hrtech_marketplace_bonus
    * distributed_inference_bonus
    * writing_detail_modifier
    * market_validation_modifier
    * legitimacy_modifier
    * profile_completeness_modifier
    * offer_acceptance_modifier
    * work_mode_modifier
)

# Apply minimum technical evidence gate
# A candidate with zero technical evidence AND near-zero skill trust density
# should NEVER appear in the top 100, regardless of their semantic similarity score.
# This blocks plain-language Tier 5 non-technical candidates.
evidence_total = (
    features["retrieval_count"] +
    features["ranking_count"] +
    features["matching_count"] +
    features["evaluation_count"] +
    features["production_count"]
)
no_evidence_mask = (evidence_total == 0) & (features["skill_trust_density"] < 0.10)
final_scores = np.where(no_evidence_mask, 0.0, final_scores)

# Apply honeypot hard exclusion LAST
final_scores = np.where(honeypot_mask, 0.0, final_scores)
```

### 9.5 Selecting and ranking the top 100

**Deterministic top-100 selection:**

```python
features = features.copy()
features["raw_score"] = final_scores
features["score"] = features["raw_score"].round(4)

# Official validator checks monotonicity and candidate_id ascending for equal
# written scores, so sort by the rounded score that will actually be written.
top_n = 100
if len(features) < 100:
    # Sandbox/demo mode for uploaded samples below 100 candidates.
    # Official submission mode must always have >=100 and output exactly 100 rows.
    top_n = len(features)

top_100_features = (
    features
    .sort_values(["score", "candidate_id"], ascending=[False, True], kind="mergesort")
    .head(top_n)
    .copy()
)
top_100_features["rank"] = range(1, len(top_100_features) + 1)
```

---

## 10. Phase 6 — Dynamic Reasoning Generator

### 10.1 The requirement

The submission spec Stage 4 review checks 10 randomly sampled reasoning entries for: specific facts from the candidate's profile, connection to JD requirements, acknowledgment of gaps, no hallucinated claims, structural variation across entries, and rank-appropriate tone.

Template-based reasoning fails all five of these checks simultaneously. The reasoning generator must build each string from the candidate's actual evidence data so that the structure itself varies based on what evidence exists.

### 10.2 The generator function

This lives in `src/explainer.py`.

```python
def generate_dynamic_reasoning(row: dict) -> str:
    """
    row: a dict with keys matching the parquet column names.
    
    IMPORTANT: Reads from the snippet string columns saved in the parquet:
        - production_snippet  (string, may be "")
        - retrieval_snippet   (string, may be "")
        - ranking_snippet     (string, may be "")
        - evaluation_snippet  (string, may be "")
    
    Do NOT pass list arrays — those are not saved in the parquet.
    Returns: 1-2 sentence reasoning string
    """
    parts = []
    yoe = row["yoe"]
    title = row["current_title"]
    rank = row["rank"]
    
    # --- Primary technical claim: structured from actual evidence snippets ---
    prod = str(row.get("production_snippet", "") or "").strip()
    retr = str(row.get("retrieval_snippet", "") or "").strip()
    rank_ev = str(row.get("ranking_snippet", "") or "").strip()
    eval_ev = str(row.get("evaluation_snippet", "") or "").strip()
    
    if prod:
        parts.append(
            f"{yoe:.0f} years applied ML; "
            f"production deployment evidence: '{prod}'"
        )
    elif retr:
        parts.append(
            f"{yoe:.0f} years with hands-on retrieval/search infrastructure: '{retr}'"
        )
    elif rank_ev:
        parts.append(
            f"{yoe:.0f} years including ranking pipeline work: '{rank_ev}'"
        )
    elif eval_ev:
        parts.append(
            f"{yoe:.0f} years; evaluation framework evidence: '{eval_ev}'"
        )
    else:
        # No evidence — must acknowledge gap honestly (Stage 4 check: honest concerns)
        parts.append(
            f"{yoe:.0f}-year {title}; "
            f"no explicit retrieval or ranking deployment evidence found in career descriptions"
        )
    
    # --- Secondary: product company context (with JD nuance) ---
    product_ratio = row["product_ratio"]
    if product_ratio >= 0.8:
        parts.append("Predominantly product-company background.")
    elif product_ratio <= 0.2:
        # Per JD: consulting-only is a noted concern, but not automatic disqualifier
        # if product experience was present at some point
        parts.append("Predominantly consulting/services background — noted gap per JD.")
    
    # --- Tertiary: behavioral availability signals ---
    notice = row.get("notice_period_days", 60)
    response_rate = row.get("recruiter_response_rate", 0.5)
    interview_rate = row.get("interview_completion_rate", 0.5)
    
    if notice <= 30 and response_rate >= 0.70:
        parts.append(
            f"Strong availability: {notice}-day notice, "
            f"{int(response_rate * 100)}% recruiter response rate."
        )
    elif notice > 90:
        parts.append(
            f"Note: {notice}-day notice period is above the company's preferred threshold."
        )
    elif response_rate < 0.30:
        parts.append(
            f"Caution: {int(response_rate * 100)}% recruiter response rate suggests limited reachability."
        )
    
    # --- MANDATORY: Cap at 2 sentences. README and submission spec both say "1-2 sentence" ---
    # Take only the first 2 parts. If parts[0] is the only entry, return it alone.
    # Never return more than 2 sentences.
    capped_parts = parts[:2]
    return " ".join(capped_parts)
```

### 10.3 Why this avoids the template problem

The structure varies because the conditional chain picks a different opening sentence depending on what evidence exists. A candidate with production evidence gets a different first sentence structure than a candidate with only retrieval evidence or a candidate with no evidence at all. The secondary and tertiary clauses are also conditional — a candidate at a product company gets different text than a consulting-heavy candidate. The resulting 100 reasoning strings will have meaningfully different structures, not just different values substituted into the same template.

### 10.4 Hallucination prevention

The generator only references:
- `yoe` from the parquet (directly from profile.years_of_experience)
- `current_title` from the parquet (directly from profile.current_title)
- Evidence strings from `production_snippet`, `retrieval_snippet`, `ranking_snippet`, and `evaluation_snippet` — these are raw substrings extracted verbatim from the candidate's career history descriptions, not generated text
- `product_ratio` from the parquet (computed deterministically from career_history)
- `notice_period_days` and `recruiter_response_rate` from the parquet (directly from redrob_signals)

Nothing in the reasoning is invented. Every claim maps directly to a field in the parquet, which maps directly to a field in the original candidate record.

---

## 11. Phase 7 — Submission Compliance & Output

### 11.1 Pre-write validation

Before writing the CSV, run these checks in order. If any check fails, raise an exception with a clear error message. Do not write a broken file.

**Check 1 — Row count:**  
For official submission mode, assert `len(top_100) == 100`. For sandbox/demo uploads with fewer than 100 input candidates, output `len(input_candidates)` rows and skip the official validator because `validate_submission.py` is intentionally strict for final submissions.

**Check 2 — Unique candidate IDs:**  
Assert `len(set(top_100["candidate_id"])) == 100`. No duplicates.

**Check 3 — Rank completeness:**  
For official submission mode, assert `set(top_100["rank"]) == set(range(1, 101))`. For sandbox/demo mode with N < 100, assert `set(top_100["rank"]) == set(range(1, N + 1))`.

**Check 4 — Score monotonicity:**  
Sort by rank. Assert that `top_100["score"].values` is non-increasing. Allow ties (equal scores are acceptable). Reject only if a lower rank has a strictly higher score.

**Check 5 — Tie-breaking:**  
For any group of candidates with identical written scores, sort that group by `candidate_id` ascending (alphabetical). Reassign ranks within the group accordingly. Use the rounded 4-decimal score for this check because the validator reads the CSV text, not the raw float.

**Check 6 — Candidate ID format:**  
Assert all IDs match `^CAND_[0-9]{7}$`.

**Check 7 — Candidate IDs exist in the input pool:**  
Assert every output `candidate_id` belongs to the `input_ids` set loaded from `--candidates`. This is mandatory because every submitted ID must exist in the released candidate file and sandbox output must rank only the uploaded sample.

**Check 8 — Score differentiation:**  
Assert there is more than one distinct score after rounding. The submission spec lists "all scores set to the same value" as a common rejection because it proves the model is not differentiating candidates.

### 11.2 Writing the CSV

```python
import csv

output_rows = []
for _, row in top_100.sort_values("rank").iterrows():
    output_rows.append({
        "candidate_id": row["candidate_id"],
        "rank": int(row["rank"]),
        "score": round(float(row["score"]), 4),
        "reasoning": generate_dynamic_reasoning(row.to_dict())
    })

with open(args.out, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f, 
        fieldnames=["candidate_id", "rank", "score", "reasoning"],
        quoting=csv.QUOTE_MINIMAL
    )
    writer.writeheader()
    writer.writerows(output_rows)
```

Important: Use UTF-8 encoding and `csv.DictWriter` with `quoting=csv.QUOTE_MINIMAL`. Do not use pandas `.to_csv()` for the final output — it can produce unexpected quoting behavior with the reasoning column.

The reproducibility command may write `submission.csv`, as required by the README example:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

For portal upload, rename/copy the final CSV to the registered participant ID filename, e.g. `team_xxx.csv`, because the submission spec requires the uploaded CSV filename to be the team's registered participant ID with `.csv` extension.

### 11.3 Running the validator

After writing the file, run the provided validator:

```bash
python validate_submission.py team_xxx.csv
```

Run the validator on the exact CSV file you will upload. This must print `Submission is valid.` with no errors before you submit.

---

## 12. Data Field Reference

This section maps every field used in the scoring formulas to its exact location in the candidate JSON schema. Use this as the definitive lookup during implementation.

| Feature Used | JSON Path | Type | Notes |
|---|---|---|---|
| years_of_experience | `profile.years_of_experience` | float | 0-50 |
| current_title | `profile.current_title` | string | |
| headline | `profile.headline` | string | Used in recall and semantic text |
| summary | `profile.summary` | string | Used in recall and semantic text |
| current_company | `profile.current_company` | string | |
| current_company_size | `profile.current_company_size` | string | Do not use alone for product/consulting classification |
| current_industry | `profile.current_industry` | string | Used in recall and product/industry signals |
| location | `profile.location` | string | City, region format |
| country | `profile.country` | string | |
| skill name | `skills[i].name` | string | |
| skill proficiency | `skills[i].proficiency` | string | beginner/intermediate/advanced/expert |
| skill endorsements | `skills[i].endorsements` | int | ≥0 |
| skill duration_months | `skills[i].duration_months` | int | ≥0 |
| career title | `career_history[i].title` | string | Used in recall and semantic text |
| career description | `career_history[i].description` | string | |
| career company | `career_history[i].company` | string | |
| career industry | `career_history[i].industry` | string | |
| career duration | `career_history[i].duration_months` | int | |
| education tier | `education[i].tier` | string | tier_1 to tier_4, unknown |
| assessment scores | `redrob_signals.skill_assessment_scores` | dict | skill_name → 0-100 |
| last active | `redrob_signals.last_active_date` | date string | YYYY-MM-DD |
| open to work | `redrob_signals.open_to_work_flag` | bool | |
| profile views | `redrob_signals.profile_views_received_30d` | int | ≥0 |
| applications submitted | `redrob_signals.applications_submitted_30d` | int | ≥0 |
| response rate | `redrob_signals.recruiter_response_rate` | float | 0.0-1.0 |
| average response time | `redrob_signals.avg_response_time_hours` | float | ≥0 |
| notice period | `redrob_signals.notice_period_days` | int | 0-180 |
| github score | `redrob_signals.github_activity_score` | float | -1 (no GitHub) to 100 |
| search appearances | `redrob_signals.search_appearance_30d` | int | ≥0 |
| saved by recruiters | `redrob_signals.saved_by_recruiters_30d` | int | ≥0 |
| interview completion | `redrob_signals.interview_completion_rate` | float | 0.0-1.0 |
| offer acceptance | `redrob_signals.offer_acceptance_rate` | float | -1 (no history) to 1.0 |
| willing to relocate | `redrob_signals.willing_to_relocate` | bool | |
| preferred work mode | `redrob_signals.preferred_work_mode` | string | remote/hybrid/onsite/flexible |
| profile completeness | `redrob_signals.profile_completeness_score` | float | 0-100 |
| total endorsements | `redrob_signals.endorsements_received` | int | ≥0 |
| verified email | `redrob_signals.verified_email` | bool | |
| verified phone | `redrob_signals.verified_phone` | bool | |
| linkedin connected | `redrob_signals.linkedin_connected` | bool | |

---

## 13. Known Traps & How Each Module Handles Them

### Trap 1 — Keyword stuffers with no career backing

**What they look like:** Candidate has 15 AI skills listed — embeddings, FAISS, Pinecone, transformers, NLP — all with high endorsement counts, but career history shows roles at a logistics company doing general software work. The descriptions say nothing about retrieval or ML.

**Which modules catch them:**
- Skill Trust Scorer: CorroborationFlag will be 0 for all the AI skills because none of them appear in the career descriptions. The trust scores will be based only on endorsements and duration, not corroboration. Overall skill_trust_density will be moderate at best.
- Regex-first Evidence Extractor: retrieval_count, ranking_count, evaluation_count, production_count will all be 0 or near-0 because the career descriptions have no relevant content.
- S_Tech will be near 0 for these candidates.

### Trap 2 — Honeypots with impossible timelines

**What they look like:** Candidate has 3 years of total experience but claims 60 months (5 years) of Python usage. Or claims "expert" in FAISS with a skill_assessment_score of 15. Or claims 8 years at a company that was founded 3 years ago. Or claims "expert" proficiency in 10 skills with 0 years of usage.

**Which modules catch them:**
- Profile Consistency Hard Gate (Section 7.2): `is_honeypot()` returns True for:
  - Any skill `duration_months > (yoe * 12) + 6`
  - Any expert/advanced skill with `duration_months == 0`
  - Any expert/advanced skill with assessment score < 40
  - Any single career role with `duration_months > (yoe * 12) + 12`
- Result: `final_score = 0.0` — excluded from top 100 entirely. This is NOT a soft penalty.

### Trap 3 — Plain-language genuine fits

**What they look like:** Real strong candidate who built a vector search system but wrote "built a document retrieval system" in their career description instead of "deployed FAISS-based dense retrieval." The embedding similarity might be moderate; the keyword match might miss them.

**Which modules protect against false negatives:**
- Retriever A (offline career semantic score) will still score them reasonably because the semantic content of "document retrieval system" is close to the JD vector — not perfectly, but close enough to make the top 1,000.
- Retriever B (skill density) will catch them if their skills list includes Python, NLP, or ML-adjacent terms with reasonable trust scores.
- The union of three retrievers means a candidate only needs to be caught by one.

### Trap 4 — Behavioral twins

**What they look like:** Two candidates with nearly identical skills and career profiles. One logged in yesterday, has 85% recruiter response rate, and is open to work. The other logged in 7 months ago, has 4% response rate, and is not marked open to work.

**Which module separates them:**
- S_Engagement: The active candidate scores 0.40 × 1.0 + 0.40 × 0.85 + 0.20 × 1.0 = 0.94. The inactive candidate scores 0.40 × 0.1 + 0.40 × 0.04 + 0.20 × 0.0 = 0.056. A nearly 17× difference in engagement score will produce a meaningful rank separation even if all other components are identical.

### Trap 5 — Consulting-only candidates with good skill profiles

**What they look like:** Candidate spent entire career at TCS, Infosys, and Wipro. Has genuinely good skills and decent career descriptions about building data pipelines. But has zero product company experience.

**Which module handles them:**
- Product Exposure Ratio: product_ratio = 0.0. Product modifier = 0.50 + 0.50 × 0.0 = 0.50. Their final score is halved relative to an equivalent candidate with full product company background. This is a soft penalty, not elimination — a consulting candidate with dramatically better technical evidence could still rank above a product-company candidate with weak evidence.

### Trap 6 — CV/Speech specialists without NLP/IR

**What they look like:** Candidate has strong background in computer vision or speech recognition. Good career history. But no retrieval, ranking, or NLP background.

**Which modules handle them:**
- S_Tech: Their regex evidence extraction returns empty or near-empty counts for retrieval and ranking evidence. Production evidence might exist for CV systems, but low retrieval and ranking counts drag the total down.
- S_Semantic: The offline career semantic score for a CV specialist will usually be lower because the JD vector is dense with retrieval and ranking vocabulary.
- They won't be completely suppressed (they are engineers with real skills) but they should naturally fall below genuine retrieval/NLP candidates.

---

## 14. Phase 8 — Sandbox Deployment (MANDATORY — Stage 1)

> **README says:** "Sandbox link is required — a working hosted environment (HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, or Binder) where your ranker can be run on a small sample. See Section 10.5 for what counts as a valid sandbox."

**This is not optional.** Submissions without a valid sandbox link are flagged at Stage 1.

### 14.1 Recommended platform: HuggingFace Spaces (Gradio)

Use a **HuggingFace Space** with the **Gradio** SDK. It is free, supports CPU-only compute, and allows file uploads. This is the lowest-friction option for a Python ranker.

### 14.2 What the sandbox must do (per submission spec Section 10.5)

1. Accept a small candidate sample (≤100 candidates) as input — via file upload
2. Run your ranking system end-to-end
3. Produce a ranked CSV output for download
4. Complete within ≤5 minutes on CPU

The sandbox does NOT need to handle the full 100K pool.

### 14.3 Sandbox architecture

The sandbox runs `rank.py` exactly. The key constraint: `artifacts/` must be pre-loaded in the Space.

**Option A (recommended): Commit artifacts to the Space**
- Pre-compute the full offline artifacts locally (parquet, embeddings, weights)
- Upload them to the HuggingFace Space as part of the repo
- The Space has 50GB storage (LFS) — artifacts should be ~2-3 GB total
- The sandbox then runs: `python rank.py --candidates uploaded_file.jsonl --out output.csv`

**Option B: On-the-fly for sandbox only**
- Since rank.py now has the on-the-fly extractor for unknown candidate IDs, the sandbox can work without pre-computed artifacts
- The sandbox runs the fast regex extractor on uploaded candidates
- NOTE: This produces slightly different scores than the full offline pipeline — acceptable for sandbox verification

### 14.4 Sandbox Gradio app (`app.py`)

```python
import gradio as gr
import subprocess
import os
import tempfile

def rank_candidates(candidate_file):
    """
    Gradio interface function.
    Accepts uploaded JSONL file, runs rank.py, returns CSV.
    """
    if candidate_file is None:
        return None, "No file uploaded."
    
    # Write uploaded file to temp location
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="wb") as tmp_in:
        tmp_in.write(candidate_file)  # Gradio passes bytes
        input_path = tmp_in.name
    
    output_path = input_path.replace(".jsonl", "_ranked.csv")
    
    # Run rank.py
    result = subprocess.run(
        ["python", "rank.py", "--candidates", input_path, "--out", output_path],
        capture_output=True, text=True, timeout=300  # 5 minute limit
    )
    
    if result.returncode != 0:
        return None, f"Error: {result.stderr}"
    
    return output_path, "Ranking complete."

demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.File(label="Upload candidate JSONL file (≤100 candidates)"),
    outputs=[
        gr.File(label="Download ranked CSV"),
        gr.Text(label="Status")
    ],
    title="Redrob Candidate Ranker — Sandbox",
    description="Upload a JSONL file of candidates (≤100). Downloads a ranked CSV."
)

demo.launch()
```

### 14.5 Pre-deployment checklist

- [ ] `all-MiniLM-L6-v2` model is in HuggingFace cache in the Space (pre-download in setup)
- [ ] `artifacts/` directory with parquet, embeddings, and weights is in the Space
- [ ] `requirements.txt` lists all dependencies with pinned versions
- [ ] Test the sandbox with `sample_candidates.json` (first 50 candidates); output may contain 50 rows because this is demo mode
- [ ] Test official validation with a 100+ candidate input and verify the output CSV passes `validate_submission.py`
- [ ] Verify the sandbox completes in under 5 minutes

---

## 15. Phase 9 — Git History & Submission Checklist

> **Submission spec Stage 4 says:** "Git history authenticity (real iteration vs single dump)."

### 15.1 Git commit strategy — CRITICAL for Stage 4

**Stage 4 manual review explicitly checks that your git history shows real iteration, not a single dump.**

This means: you must commit incrementally as you build each phase. Do NOT code everything, then make one commit.

**Required commit cadence:**
1. After Phase 0 (validation harness) → commit
2. After Phase 1 (streaming filter) → commit
3. After Phase 2 (tri-vector retrieval funnel) → commit
4. After Phase 3 (feature factory complete) → commit
5. After Phase 4 (weight optimization) → commit with the actual weights found
6. After rank.py works end-to-end → commit
7. After validation — `validate_submission.py` passes → commit tagged `v1-submission`
8. After any tuning iteration → commit
9. Final submission version → commit tagged `final-submission`

**Bad git history (Stage 4 fail):**
```
commit abc123 — "Add everything" (1 commit, all files)
```

**Good git history (Stage 4 pass):**
```
commit f1a2b3 — "Phase 0: validation harness + 150-candidate validation set"
commit c4d5e6 — "Phase 1: streaming filter, 42K candidates in staging pool"
commit 789abc — "Phase 2: tri-vector funnel, 2,400 operational candidates"
commit def012 — "Phase 3: regex evidence extractor + full feature store"
commit 345678 — "Phase 4: weight optimization, NDCG@10=0.74 on local validation"
commit 9abcde — "rank.py: passing validate_submission.py"
commit f12345 — "Tune: raised retrieval_count weight, NDCG@10=0.79"
commit 890abc — "Final: sandbox deployed on HuggingFace Spaces"
```

### 15.2 submission_metadata.yaml — Required Fields

The final codebase must include `submission_metadata.yaml` at the repository root. The structure and keys must match the official template (`submission_metadata_template.yaml`) exactly. Pre-fill this file in your repository:

```yaml
# Redrob Hackathon — Submission Metadata
team_name: "your-team-name-here"

primary_contact:
  name: "Your Full Name"
  email: "primary@example.com"
  phone: "+91-XXXXXXXXXX"

team_members:
  - name: "Your Full Name"
    email: "primary@example.com"
    role: "ML Engineer"

github_repo: "https://github.com/YOUR_USERNAME/YOUR_REPO"
sandbox_link: "https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker"
reproduce_command: "python rank.py --candidates ./candidates.jsonl --out ./submission.csv"

compute:
  platform: "Local Linux box"
  cpu_cores: 8
  ram_gb: 16
  python_version: "3.11.4"
  os: "Ubuntu 22.04 LTS"
  uses_gpu_for_inference: false
  has_network_during_ranking: false
  pre_computation_required: true
  pre_computation_time_minutes: 180

ai_tools_used:
  - "Gemini"
  - "Cursor"

ai_usage_summary: |
  Used Gemini for code review, debugging assistance, and plan verification.
  No candidate profile data was sent to external LLMs.
methodology_summary: |
  Two-phase pipeline: offline feature factory pre-computes scalar technical,
  semantic, integrity, and behavioral features for all 100K candidates. Runtime
  rank.py loads Parquet features and scores via a 7-component formula + soft modifiers.
  Honeypots are hard-excluded. Weights optimized via scipy Nelder-Mead.

declarations:
  read_submission_spec: true
  code_is_original_work: true
  no_collusion: true
  honeypot_check_done: true
  reproduction_tested: true
```


### 15.3 Pre-submission checklist

- [ ] Ranked output CSV exists and passes official validation in full-submission mode
- [ ] `validate_submission.py team_xxx.csv` passes on the exact upload file
- [ ] Portal upload file is named with the registered participant/team ID, e.g. `team_xxx.csv`
- [ ] Visually compare first 3 lines of submission.csv against `sample_submission.csv`
- [ ] Honeypot rate in top 100 verified to be < 10% (manually check is_honeypot flags)
- [ ] All 100 rows present, ranks 1-100 each appear exactly once
- [ ] Score is non-increasing from rank 1 to rank 100
- [ ] Scores are not all identical after rounding
- [ ] Reasoning strings are 1-2 sentences (no longer)
- [ ] No candidate_id appears twice
- [ ] All candidate_ids match `^CAND_[0-9]{7}$`
- [ ] Every output candidate_id exists in the input candidates file
- [ ] Sandbox works with `sample_candidates.json`
- [ ] GitHub repo is public (or access granted to organizers)
- [ ] Repo contains the full source that produced the CSV; no hidden scripts, notebooks, or manual CSV edits are required
- [ ] Any required precomputed artifact is committed/provided, or `preprocess.py` documents how to generate it
- [ ] README.md in repo has exact command: `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`
- [ ] `requirements.txt` pinned with exact versions
- [ ] `submission_metadata.yaml` at repo root is complete
- [ ] Sandbox link is live and included in metadata
- [ ] Runtime artifacts/intermediate state used by `rank.py` stay under the 5 GB disk limit
- [ ] Architecture note or deck/PPT converted to PDF explains the pipeline, major scoring components, constraints, and why the design avoids keyword-stuffer traps
- [ ] Brief video demo shows the engine running and producing explainable ranked candidate results
- [ ] Git history shows real iteration (at least 8 commits)

---

## 16. Dependency Declarations

The `requirements.txt` must specify exact versions to ensure reproducibility:

```
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0
scikit-learn==1.5.0
sentence-transformers==3.0.1
pyarrow==16.1.0
```

`sentence-transformers` is required by `preprocess.py` for offline semantic scoring, but `rank.py` must not import it or load any transformer model at runtime. The final ranking step should use the precomputed scalar `career_semantic_score` stored in Parquet.

The total installed dependency size plus runtime artifacts should stay within the 5 GB disk/intermediate-state constraint.

---

*End of specification. Version 2.0.0.*
