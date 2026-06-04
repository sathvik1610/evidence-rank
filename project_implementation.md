# Intelligent Candidate Discovery & Ranking Engine
## Production System Specification — Version 2.0.0 (Final)

**Project:** Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge  
**Role Being Ranked For:** Senior AI Engineer, Founding Team, Redrob AI  
**Dataset:** 100,000 candidates in `candidates.jsonl.gz`  
**Output:** `submission.csv` — top 100 candidates, ranked best-fit first  
**Hard Constraints:** 5-minute wall-clock execution, CPU-only, 16 GB RAM, zero network calls during ranking

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

Read the JD not as a checklist but as a personality profile of the company's hiring logic.

**Hard requirements the system must operationalize:**

- Production experience with embeddings-based retrieval systems deployed to real users. The key word is production. Side projects, tutorials, and Kaggle notebooks do not count. The career history descriptions must contain evidence of deployment at scale.
- Production experience with vector databases or hybrid search. Same logic. It does not matter which specific database they used. It matters that they operated one in a real system.
- Strong Python with evidence of code quality, not just familiarity.
- Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation. This is a strong positive signal. Most candidates will not have it.

**Hard disqualifiers the system must penalize heavily:**

- Pure research background with no production deployment history. These candidates will have strong academic credentials, possibly good skills listings, but their career history descriptions will lack deployment evidence. The LLM extractor will return empty production_evidence arrays.
- "AI experience" consisting primarily of LangChain wrappers calling hosted APIs with under 12 months of this kind of work and no pre-LLM ML background. These candidates will have LangChain, OpenAI, LlamaIndex in their skills but shallow career history descriptions.
- Candidates whose entire career is at consulting firms — TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini. The JD names these companies explicitly. If every role in career_history maps to one of these companies with no product-company experience at all, this is a heavy penalty.
- Computer vision, speech, or robotics specialists without NLP or IR exposure. These candidates will have strong skills in OpenCV, YOLO, ASR, TTS, image classification, but shallow or absent retrieval and ranking history.
- Title-chasers with a pattern of switching companies every 12-18 months for title progression. This is detectable by examining career_history duration patterns.

**The JD's ideal candidate profile for the scoring formula:**

- 6-8 years total experience, 4-5 of which are at product companies in applied ML roles
- Has shipped at least one end-to-end ranking, search, or recommendation system at real scale
- Located in or willing to relocate to Pune or Noida
- Active on the platform — recently logged in, responds to recruiters, notice period under 30 days is strongly preferred

### 1.3 What the behavioral signals actually mean

The 23 redrob_signals fields are not profile quality metrics. They are availability and reachability metrics. A candidate with a perfect skills profile and a 3% recruiter response rate who last logged in 8 months ago is, for practical hiring purposes, not a real option. The signals should be used as a multiplicative modifier on top of the skill-and-experience score, not as a primary ranking signal.

The most important behavioral signals for this role are:
- `last_active_date` — recency of platform activity
- `open_to_work_flag` — explicit availability signal
- `recruiter_response_rate` — reachability
- `notice_period_days` — time to hire (the JD explicitly prefers sub-30 days)
- `interview_completion_rate` — reliability once engaged
- `github_activity_score` — for an AI engineer role, active GitHub is a positive signal; -1 means no GitHub linked, which is neither positive nor negative

### 1.4 The trap categories and how each one fails a naive system

**Keyword stuffers.** These candidates list every AI keyword in their skills array but their career history descriptions don't back it up. A naive embedding system ranks them highly because their skill text is dense with target tokens. The Skill Trust Scorer catches them by requiring corroboration from career descriptions and checking duration and endorsements.

**Honeypots with impossible timelines.** These candidates have structural impossibilities: a skill listed with duration_months greater than their entire career length, expert proficiency in a skill with 0 months listed, 8 years of experience at a company founded 3 years ago. The Profile Consistency Engine catches these.

**Plain-language genuine fits.** These are real strong candidates who described their work without using the fashionable keywords — they might say "built a recommendation system" rather than "deployed dense retrieval with FAISS." A system that only does keyword or embedding matching on skills text will miss them. The career embedding retriever and the LLM extractor catch them by reading the full career description text.

**Behavioral twins.** Two candidates with near-identical skill profiles and career histories, but one hasn't logged in for 6 months and has a 5% response rate while the other is actively seeking. They should not rank equally. The platform engagement score separates them.

---

## 2. System Architecture Overview

The system is divided into two strictly separated execution environments:

**The Offline Factory** (no time limit, run once before submission):
- Streams all 100,000 candidates and filters to a high-signal pool of 40,000-50,000
- Runs tri-vector retrieval to narrow to approximately 2,500 operational candidates
- Runs the full feature factory on those 2,500: Skill Trust scoring, Profile Consistency scoring, LLM evidence extraction, embedding generation
- Serializes all features to `artifacts/candidate_features.parquet`
- Serializes all embeddings to `artifacts/candidate_career_embeddings.npy` and `artifacts/candidate_skills_embeddings.npy`
- Optimizes scoring weights against the local validation set
- Saves optimized weights to `artifacts/optimized_weights.json`

**The Runtime Engine** (must complete in under 5 minutes on CPU, no network):
- `rank.py` is the only script that runs at evaluation time
- Loads precomputed features and embeddings from `artifacts/`
- Runs the vectorized 7-component scoring formula in NumPy
- Applies soft modifiers
- Generates reasoning strings
- Validates and writes the final CSV

The two environments share no code path during ranking. The offline factory can use any compute it needs. The runtime engine touches nothing that requires network or heavy computation.

---

## 3. Repository Layout

The final repository must match this structure exactly. No other layout will pass Stage 3 code review.

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
    ├── candidate_career_embeddings.npy
    ├── candidate_skills_embeddings.npy
    └── optimized_weights.json
```

**What each file does:**

`rank.py` — The single entry point for ranking. Takes `--candidates` and `--out` as CLI arguments. Loads precomputed artifacts, scores candidates, generates reasoning, writes the output CSV. Must complete in under 5 minutes.

`preprocess.py` — Runs the full offline pipeline: streaming filter → retrieval funnel → feature factory → embedding generation. Writes all artifacts. Can take hours; this is not constrained by the 5-minute rule.

`train_weights.py` — Loads the validation set, runs the scoring formula with variable weights, optimizes using SciPy Nelder-Mead, saves the result to `artifacts/optimized_weights.json`.

`src/indexer.py` — The streaming token filter. Reads `candidates.jsonl.gz` line by line and writes `artifacts/high_signal_pool.jsonl`.

`src/features.py` — Contains the Skill Trust Score formula, the Profile Consistency Score formula, and the Product Company Classifier.

`src/inference.py` — Runs the local LLM (Llama-3-8B-Q4 via llama-cpp-python) over the 2,500 operational candidates and extracts evidence arrays.

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

NDCG implementation uses the standard formula with tier as the relevance score (tier 3 = 3, tier 2 = 2, tier 1 = 0, tier 0 = 0). MAP is computed across all tier-3 candidates as positives.

The harness function signature:

```python
def evaluate_ranking(ranked_candidate_ids: list[str], validation_set: list[dict]) -> dict:
    """
    ranked_candidate_ids: ordered list of all 100 candidate IDs from your submission, rank 1 first
    validation_set: loaded from metadata/validation_set.json
    Returns: dict with keys ndcg_10, ndcg_50, map, p_10, composite
    """
```

---

## 5. Phase 1 — Streaming Ingestion & Recall Filter

### 5.1 What this phase does

Reads `candidates.jsonl.gz` line by line without loading the full 465 MB into memory. For each candidate, checks three fields against a keyword index. If the candidate has zero matching tokens across all three fields, they are dropped. Everyone else goes to the staging file.

The purpose is not to find good candidates. It is to eliminate candidates who are categorically unrelated to technical roles — accountants, lawyers, civil engineers, teachers. This is a recall filter, not a precision filter. When in doubt, keep the candidate.

### 5.2 Fields to check

Check these three fields only, after lowercasing:
- `profile.headline`
- `profile.current_title`
- `education[0].field_of_study` (use the first education entry only; if the array is empty, skip this field)

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

with gzip.open("candidates.jsonl.gz", "rt", encoding="utf-8") as f_in, \
     open("artifacts/high_signal_pool.jsonl", "w", encoding="utf-8") as f_out:
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

Takes the 40,000-50,000 staging candidates and narrows them to approximately 2,500 using three independent retrievers running in parallel. The union of all three retrievers' outputs becomes the operational pool for the expensive feature factory.

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
    parts = [candidate["profile"]["current_title"]]
    for role in candidate["career_history"]:
        parts.append(role["title"])
        parts.append(role["description"])
    return " ".join(parts)
```

Encode all career texts in batches of 256 using `model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)`.

Compute cosine similarity against `jd_vector` using NumPy dot product (since vectors are already normalized, dot product equals cosine similarity).

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

This union will typically contain 2,000 to 3,000 unique candidates. Load all of them from the staging file into memory — they now become the operational pool for the feature factory.

If the operational pool exceeds 4,000 candidates, the LLM extraction phase will take too long. In that case, score each candidate using the quick_skill_density function and keep only the top 3,000 by that score.

**Recall verification — do this before running the feature factory:**

After building the operational pool, manually inspect 10-15 candidates from your validation set's Tier 3 group and confirm they are all present in the pool. If any Tier 3 candidate is missing, trace which retriever should have caught them and why it didn't. A missing Tier 3 candidate at this stage is unrecoverable — they will never appear in the final top 100. Fix the retriever thresholds before proceeding. This check takes 10 minutes and prevents the most damaging silent failure in the entire pipeline.

---

## 7. Phase 3 — Offline Feature Factory

This phase processes every candidate in the operational pool (approximately 2,500 candidates) and extracts the features that the runtime scoring engine will use. This is the computationally expensive phase. It has no time limit.

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

### 7.2 Component B — Profile Consistency Scorer

This component detects structural anomalies that indicate a honeypot or fabricated profile.

**The formula:**

```
Consistency Score = 1.0 - clip(
    0.35 × N_timeline + 0.35 × N_assessments + 0.30 × N_titles,
    0.0, 1.0
)
```

**Parameter definitions:**

`N_timeline` — binary counter, increments by 1 if ANY single skill's `duration_months` strictly exceeds `(years_of_experience × 12) + 6`. The +6 is a grace buffer. This detects the honeypot pattern where a candidate with 4 years of experience claims 60 months of Python usage. Check every skill in the array; if any one violates the constraint, N_timeline = 1. It does not increment past 1.

`N_assessments` — binary counter, increments by 1 if any skill labeled `"expert"` or `"advanced"` has a corresponding entry in `redrob_signals.skill_assessment_scores` with a value below 40. Only check skills that have a corresponding assessment entry — if the skill name does not appear in skill_assessment_scores, skip it. If no such violation exists, N_assessments = 0.

`N_titles` — binary counter, increments by 1 if `years_of_experience > 6` AND `current_title` contains any of these substrings (case-insensitive): "junior", "associate", "intern", "trainee", "entry". This catches profiles where someone claims 8 years of experience but holds a junior title — a structural inconsistency.

**The hard suppression threshold:**

If `Consistency Score < 0.30`, this candidate is almost certainly a honeypot. At the scoring stage, apply a multiplier of 0.01 to their final score to push them below rank 100 without breaking array sorting operations.

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

### 7.4 Component D — Local LLM Evidence Extractor

This is the most computationally expensive component. Run it on the operational pool of ~2,500 candidates using a local quantized LLM.

**Model specification:**

Use `Llama-3-8B-Instruct-Q4_K_M.gguf` via the `llama-cpp-python` library. Load with `n_ctx=2048` and `n_threads` equal to your available CPU cores.

Download the model from HuggingFace before running. The model file is approximately 4.5 GB. Store it at `models/llama-3-8b-instruct-q4_k_m.gguf`.

**Benchmark before committing to the LLM path.** Run the LLM extractor on exactly 100 candidates and time it. If 100 candidates takes more than 2 minutes, the full 2,500-candidate run will exceed 50 minutes — unacceptable even for the offline phase. In that case, switch entirely to the regex fallback. Do not attempt a hybrid where some candidates get LLM extraction and others get regex — this creates inconsistent feature scales across the parquet. Pick one path and apply it uniformly to all candidates.

**Fallback — Regex Evidence Extractor (use this if LLM benchmark fails the 2-minute threshold):**

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

For each pattern group, count the number of distinct pattern matches in the concatenated career description text. Store as `retrieval_count`, `ranking_count`, `evaluation_count`, `production_count`.

**The LLM prompt (if using LLM path):**

```
You are a factual information extractor. Read the career history text below and extract only facts that are explicitly stated. Do not infer, guess, or summarize. Return JSON only with no explanation.

CAREER HISTORY:
{career_text}

Extract phrases that are EXPLICITLY STATED in the text for each category. If nothing is explicitly stated for a category, return an empty array. Never invent examples.

Return this exact JSON structure:
{
  "retrieval_evidence": [],
  "ranking_evidence": [],
  "evaluation_evidence": [],
  "production_evidence": []
}
```

Critical implementation notes for the LLM path:
- Set temperature to 0.0 for deterministic output
- Set max_tokens to 400 — the output should be compact JSON
- Strip any markdown code fences before parsing (`json.loads` will fail on ```json wrappers)
- Wrap every LLM call in a try-except. If JSON parsing fails, fall back to the regex extractor for that candidate
- Never pass more than 1,200 tokens of career text to the model. If the career text is longer, truncate from the end

**The output feature:**

Regardless of which path was used (LLM or regex), the feature stored in the parquet is a single integer per evidence category:

```
retrieval_count = len(retrieval_evidence) or regex_match_count
ranking_count = len(ranking_evidence) or regex_match_count
evaluation_count = len(evaluation_evidence) or regex_match_count
production_count = len(production_evidence) or regex_match_count
```

Also store the first snippet from `production_evidence` (or `retrieval_evidence` if production is empty) as `primary_evidence_snippet` — a string column in the parquet for use in reasoning generation.

### 7.5 Component E — Embedding Generation

Generate two embedding vectors per candidate using `all-MiniLM-L6-v2`.

**Career embedding:**

Build the career text string exactly as in Section 6.4 (current_title + all role titles + all role descriptions concatenated). Truncate to 512 tokens if needed.

**Trusted skills embedding:**

Build a string containing only the names of skills whose individual Skill Trust Score (computed in Component A) is strictly greater than 0.5. Concatenate with spaces. If no skills pass the threshold, use an empty string — the embedding will be a near-zero vector, which is appropriate.

```python
trusted_skills_text = " ".join(
    skill["name"] for skill in candidate["skills"]
    if compute_skill_trust_score(skill, career_text) > 0.5
)
```

**Serialization:**

After processing all operational candidates, stack all career embeddings into a single NumPy array of shape `(N, 384)` and save to `artifacts/candidate_career_embeddings.npy`. Save the corresponding candidate IDs as a separate array to `artifacts/embedding_candidate_ids.npy` so the runtime engine can look up which row corresponds to which candidate.

Do the same for trusted skills embeddings → `artifacts/candidate_skills_embeddings.npy`.

### 7.6 The Parquet Feature Store

After all components run, write `artifacts/candidate_features.parquet` with one row per candidate in the operational pool. The columns are:

| Column | Type | Source |
|--------|------|--------|
| candidate_id | string | profile |
| yoe | float | profile.years_of_experience |
| current_title | string | profile.current_title |
| current_company | string | profile.current_company |
| location | string | profile.location |
| country | string | profile.country |
| skill_trust_density | float | Component A |
| raw_skill_trust_sum | float | Component A |
| consistency_score | float | Component B |
| product_ratio | float | Component C |
| retrieval_count | int | Component D |
| ranking_count | int | Component D |
| evaluation_count | int | Component D |
| production_count | int | Component D |
| primary_evidence_snippet | string | Component D |
| notice_period_days | int | redrob_signals |
| recruiter_response_rate | float | redrob_signals |
| open_to_work_flag | bool | redrob_signals |
| last_active_date | string | redrob_signals |
| interview_completion_rate | float | redrob_signals |
| github_activity_score | float | redrob_signals |
| offer_acceptance_rate | float | redrob_signals |
| willing_to_relocate | bool | redrob_signals |
| preferred_work_mode | string | redrob_signals |

---

## 8. Phase 4 — Weight Optimization Loop

### 8.1 What this does

Instead of hardcoding the 7 component weights, use the validation set to find the weight vector that maximizes the local composite NDCG score. This is an offline step that runs once after the feature factory completes and before rank.py is finalized.

### 8.2 The scoring formula to be optimized

The weights w1 through w7 correspond to:

```
w1 = weight for S_Tech (Technical Strength)
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
        "w1_tech": float(optimized_weights[0]),
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

This is the script that runs at evaluation time. It must complete in under 5 minutes on CPU with 16 GB RAM and no network access.

### 9.1 Startup and loading

```python
import argparse
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from src.explainer import generate_dynamic_reasoning

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    # Load precomputed artifacts
    features = pd.read_parquet("artifacts/candidate_features.parquet")
    career_embeddings = np.load("artifacts/candidate_career_embeddings.npy", mmap_mode="r")
    embedding_ids = np.load("artifacts/embedding_candidate_ids.npy", allow_pickle=True)
    weights = json.load(open("artifacts/optimized_weights.json"))
    
    # Build JD vector (model runs on CPU, 384-dim, fast)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_vector = model.encode(JD_QUERY_TEXT, normalize_embeddings=True)
```

### 9.2 The 7 scoring components

Compute each component as a NumPy array over all candidates in the features dataframe. All computations are vectorized — no Python loops over candidates.

**S_Tech — Technical Strength**

```python
evidence_sum = (
    features["retrieval_count"] + 
    features["ranking_count"] + 
    features["evaluation_count"] + 
    features["production_count"]
)
S_Tech = np.log1p(evidence_sum.values)
S_Tech = S_Tech / S_Tech.max()  # normalize to 0-1
```

**S_Trust — Skill Trust Density**

```python
S_Trust = features["skill_trust_density"].values.clip(0, 1)
```

**S_Semantic — Career Embedding Cosine Similarity**

```python
# career_embeddings is already normalized during generation
# jd_vector is already normalized
S_Semantic = career_embeddings @ jd_vector  # shape: (N,)
S_Semantic = (S_Semantic + 1) / 2  # rescale from [-1,1] to [0,1]
```

**S_Integrity — Profile Consistency Score**

```python
S_Integrity = features["consistency_score"].values.clip(0, 1)
```

**S_Engagement — Platform Engagement**

```python
# RecencyScore: days since last_active_date
from datetime import date
today = date.today()

def compute_recency(last_active_str):
    if pd.isna(last_active_str):
        return 0.1
    days_inactive = (today - date.fromisoformat(last_active_str)).days
    if days_inactive <= 30:
        return 1.0
    elif days_inactive <= 60:
        return 0.8
    elif days_inactive <= 90:
        return 0.6
    elif days_inactive <= 180:
        return 0.4
    else:
        return 0.1

recency_scores = features["last_active_date"].apply(compute_recency).values
response_rates = features["recruiter_response_rate"].fillna(0.5).values
open_to_work = features["open_to_work_flag"].astype(float).values

S_Engagement = (
    0.40 * recency_scores +
    0.40 * response_rates +
    0.20 * open_to_work
)
```

**S_Seniority — Experience Alignment**

```python
yoe = features["yoe"].values
# Gaussian centered at 7.0, sigma = 2.0
# Peaks at exactly 7 years, drops smoothly for shorter or longer
S_Seniority = np.exp(-((yoe - 7.0) ** 2) / (2 * (2.0 ** 2)))
```

**S_Proximity — Geographic Alignment**

```python
PUNE_NOIDA_CITIES = {
    "pune", "noida", "greater noida", "delhi", "new delhi", "gurugram", 
    "gurgaon", "faridabad", "ghaziabad"
}
INDIA_ADJACENT = {
    "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", 
    "kolkata", "ahmedabad", "indore", "jaipur", "chandigarh"
}

def compute_proximity(row):
    location = str(row["location"]).lower()
    country = str(row["country"]).lower()
    willing = bool(row["willing_to_relocate"])
    
    if any(city in location for city in PUNE_NOIDA_CITIES):
        return 1.0
    if willing and country == "india":
        return 0.9
    if country == "india":
        return 0.8
    if willing:
        return 0.6
    return 0.5

S_Proximity = features.apply(compute_proximity, axis=1).values
```

### 9.3 The weighted combination

```python
w = np.array([
    weights["w1_tech"],
    weights["w2_trust"],
    weights["w3_semantic"],
    weights["w4_integrity"],
    weights["w5_engagement"],
    weights["w6_seniority"],
    weights["w7_proximity"]
])

raw_scores = (
    w[0] * S_Tech +
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
```

**Notice Period Gradient:**

```python
def notice_modifier(days):
    if days <= 30:
        return 1.00
    elif days <= 60:
        return 0.85
    elif days <= 90:
        return 0.65
    else:
        return 0.30

notice_days = features["notice_period_days"].fillna(60).values
notice_modifiers = np.vectorize(notice_modifier)(notice_days)
```

**Honeypot Suppression:**

```python
consistency = features["consistency_score"].values
honeypot_multiplier = np.where(consistency < 0.30, 0.01, 1.0)
```

**Combined final score:**

```python
final_scores = raw_scores * product_modifier * notice_modifiers * honeypot_multiplier
```

### 9.5 Selecting and ranking the top 100

```python
# Get indices of top 100 by final score
top_100_idx = np.argsort(final_scores)[::-1][:100]
top_100_features = features.iloc[top_100_idx].copy()
top_100_scores = final_scores[top_100_idx]

# Assign ranks 1-100
top_100_features["rank"] = range(1, 101)
top_100_features["score"] = top_100_scores
```

---

## 10. Phase 6 — Dynamic Reasoning Generator

### 10.1 The requirement

The submission spec Stage 4 review checks 10 randomly sampled reasoning entries for: specific facts from the candidate's profile, connection to JD requirements, acknowledgment of gaps, no hallucinated claims, structural variation across entries, and rank-appropriate tone.

Template-based reasoning fails all five of these checks simultaneously. The reasoning generator must build each string from the candidate's actual evidence data so that the structure itself varies based on what evidence exists.

### 10.2 The generator function

This lives in `src/explainer.py`. The function receives a single candidate row and returns a string of 1-3 sentences. The logic uses a priority-ordered evidence chain — the first non-empty evidence type determines the primary claim structure.

```python
def generate_dynamic_reasoning(row: dict) -> str:
    """
    row: a dict with keys matching the parquet column names plus
         retrieval_evidence, ranking_evidence, evaluation_evidence,
         production_evidence (lists of strings from LLM or empty lists)
    Returns: 1-3 sentence reasoning string
    """
    parts = []
    yoe = row["yoe"]
    title = row["current_title"]
    rank = row["rank"]
    
    # --- Primary technical claim: structured from actual evidence ---
    prod = row.get("production_evidence", [])
    retr = row.get("retrieval_evidence", [])
    rank_ev = row.get("ranking_evidence", [])
    eval_ev = row.get("evaluation_evidence", [])
    
    if prod:
        parts.append(
            f"{yoe:.0f} years applied ML experience; "
            f"production-scale deployment evidence: {prod[0]}"
        )
    elif retr:
        parts.append(
            f"{yoe:.0f} years experience with hands-on retrieval infrastructure: {retr[0]}"
        )
    elif rank_ev:
        parts.append(
            f"{yoe:.0f} years experience including ranking pipeline work: {rank_ev[0]}"
        )
    elif eval_ev:
        parts.append(
            f"{yoe:.0f} years experience; evaluation framework evidence: {eval_ev[0]}"
        )
    else:
        parts.append(
            f"{yoe:.0f} years in {title} role; "
            f"no explicit retrieval or ranking deployment evidence found in career descriptions"
        )
    
    # --- Secondary: product company context ---
    product_ratio = row["product_ratio"]
    if product_ratio >= 0.8:
        parts.append("Career predominantly at product companies.")
    elif product_ratio <= 0.2:
        parts.append("Career predominantly at consulting/services firms — noted gap.")
    
    # --- Tertiary: behavioral availability signals ---
    notice = row["notice_period_days"]
    response_rate = row["recruiter_response_rate"]
    
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
    
    return " ".join(parts)
```

### 10.3 Why this avoids the template problem

The structure varies because the conditional chain picks a different opening sentence depending on what evidence exists. A candidate with production evidence gets a different first sentence structure than a candidate with only retrieval evidence or a candidate with no evidence at all. The secondary and tertiary clauses are also conditional — a candidate at a product company gets different text than a consulting-heavy candidate. The resulting 100 reasoning strings will have meaningfully different structures, not just different values substituted into the same template.

### 10.4 Hallucination prevention

The generator only references:
- `yoe` from the parquet (directly from profile.years_of_experience)
- `current_title` from the parquet (directly from profile.current_title)
- Evidence strings from `production_evidence[0]`, `retrieval_evidence[0]`, etc. — these are raw substrings extracted verbatim from the candidate's career history descriptions, not generated text
- `product_ratio` from the parquet (computed deterministically from career_history)
- `notice_period_days` and `recruiter_response_rate` from the parquet (directly from redrob_signals)

Nothing in the reasoning is invented. Every claim maps directly to a field in the parquet, which maps directly to a field in the original candidate record.

---

## 11. Phase 7 — Submission Compliance & Output

### 11.1 Pre-write validation

Before writing the CSV, run these checks in order. If any check fails, raise an exception with a clear error message. Do not write a broken file.

**Check 1 — Row count:**  
Assert `len(top_100) == 100`. If not, something went wrong in the selection step.

**Check 2 — Unique candidate IDs:**  
Assert `len(set(top_100["candidate_id"])) == 100`. No duplicates.

**Check 3 — Rank completeness:**  
Assert `set(top_100["rank"]) == set(range(1, 101))`. Every integer from 1 to 100 must appear exactly once.

**Check 4 — Score monotonicity:**  
Sort by rank. Assert that `top_100["score"].values` is non-increasing. Allow ties (equal scores are acceptable). Reject only if a lower rank has a strictly higher score.

**Check 5 — Tie-breaking:**  
For any group of candidates with identical scores, sort that group by `candidate_id` ascending (alphabetical). Reassign ranks within the group accordingly.

**Check 6 — Candidate ID format:**  
Assert all IDs match `^CAND_[0-9]{7}$`.

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

Important: Use `csv.DictWriter` with `quoting=csv.QUOTE_MINIMAL`. Do not use pandas `.to_csv()` for the final output — it can produce unexpected quoting behavior with the reasoning column.

### 11.3 Running the validator

After writing the file, run the provided validator:

```bash
python validate_submission.py submission.csv
```

This must print `Submission is valid.` with no errors before you submit.

---

## 12. Data Field Reference

This section maps every field used in the scoring formulas to its exact location in the candidate JSON schema. Use this as the definitive lookup during implementation.

| Feature Used | JSON Path | Type | Notes |
|---|---|---|---|
| years_of_experience | `profile.years_of_experience` | float | 0-50 |
| current_title | `profile.current_title` | string | |
| location | `profile.location` | string | City, region format |
| country | `profile.country` | string | |
| skill name | `skills[i].name` | string | |
| skill proficiency | `skills[i].proficiency` | string | beginner/intermediate/advanced/expert |
| skill endorsements | `skills[i].endorsements` | int | ≥0 |
| skill duration_months | `skills[i].duration_months` | int | ≥0 |
| career description | `career_history[i].description` | string | |
| career company | `career_history[i].company` | string | |
| career industry | `career_history[i].industry` | string | |
| career duration | `career_history[i].duration_months` | int | |
| education tier | `education[i].tier` | string | tier_1 to tier_4, unknown |
| assessment scores | `redrob_signals.skill_assessment_scores` | dict | skill_name → 0-100 |
| last active | `redrob_signals.last_active_date` | date string | YYYY-MM-DD |
| open to work | `redrob_signals.open_to_work_flag` | bool | |
| response rate | `redrob_signals.recruiter_response_rate` | float | 0.0-1.0 |
| notice period | `redrob_signals.notice_period_days` | int | 0-180 |
| github score | `redrob_signals.github_activity_score` | float | -1 (no GitHub) to 100 |
| interview completion | `redrob_signals.interview_completion_rate` | float | 0.0-1.0 |
| offer acceptance | `redrob_signals.offer_acceptance_rate` | float | -1 (no history) to 1.0 |
| willing to relocate | `redrob_signals.willing_to_relocate` | bool | |
| preferred work mode | `redrob_signals.preferred_work_mode` | string | remote/hybrid/onsite/flexible |

---

## 13. Known Traps & How Each Module Handles Them

### Trap 1 — Keyword stuffers with no career backing

**What they look like:** Candidate has 15 AI skills listed — embeddings, FAISS, Pinecone, transformers, NLP — all with high endorsement counts, but career history shows roles at a logistics company doing general software work. The descriptions say nothing about retrieval or ML.

**Which modules catch them:**
- Skill Trust Scorer: CorroborationFlag will be 0 for all the AI skills because none of them appear in the career descriptions. The trust scores will be based only on endorsements and duration, not corroboration. Overall skill_trust_density will be moderate at best.
- LLM Extractor / Regex Extractor: retrieval_count, ranking_count, evaluation_count, production_count will all be 0 or near-0 because the career descriptions have no relevant content.
- S_Tech will be near 0 for these candidates.

### Trap 2 — Honeypots with impossible timelines

**What they look like:** Candidate has 3 years of total experience but claims 60 months (5 years) of Python usage. Or claims "expert" in FAISS with a skill_assessment_score of 15.

**Which modules catch them:**
- Profile Consistency Engine: N_timeline fires when any skill's duration_months exceeds (years_of_experience × 12) + 6.
- N_assessments fires when an expert/advanced skill has a low assessment score.
- Consistency Score drops to 0.65 or below. If it drops below 0.30, the 0.01 honeypot suppression multiplier activates.

### Trap 3 — Plain-language genuine fits

**What they look like:** Real strong candidate who built a vector search system but wrote "built a document retrieval system" in their career description instead of "deployed FAISS-based dense retrieval." The embedding similarity might be moderate; the keyword match might miss them.

**Which modules protect against false negatives:**
- Retriever A (career embeddings) will still score them reasonably because the semantic content of "document retrieval system" is close to the JD vector — not perfectly, but close enough to make the top 1,000.
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
- S_Tech: Their LLM evidence extraction returns empty or near-empty arrays for retrieval_evidence and ranking_evidence. production_evidence might have entries about CV systems. But the S_Tech formula uses log1p of all four counts summed. Low retrieval and ranking counts drag the total down.
- S_Semantic: Career embeddings for a CV specialist will have lower cosine similarity to the JD vector, which is dense with retrieval and ranking vocabulary.
- They won't be completely suppressed (they are engineers with real skills) but they should naturally fall below genuine retrieval/NLP candidates.

---

## 14. Dependency Declarations

The `requirements.txt` must specify exact versions to ensure reproducibility:

```
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0
scikit-learn==1.5.0
sentence-transformers==3.0.1
pyarrow==16.1.0
llama-cpp-python==0.2.78
```

The `llama-cpp-python` package is optional — only include it if you are using the LLM extraction path. If using the regex fallback only, remove it from requirements.txt.

The `sentence-transformers` package will download `all-MiniLM-L6-v2` automatically on first use. For the offline factory this is fine. For `rank.py`, the model must already be cached locally — it cannot download during the no-network ranking step. Before submitting, confirm that the model is in the local HuggingFace cache by running `rank.py` once with network disabled.

The total installed dependency size should stay under 2 GB to remain within the 5 GB disk constraint.

## Additional comments (DO NOT IGNORE):

Only 3 things I would still change
1. Remove LLM extraction entirely unless benchmarking proves it is clearly superior

For this competition:

Regex + evidence patterns
>
Slow local LLM

until proven otherwise.

I would treat LLM extraction as optional, not primary.

2. Store actual evidence arrays in parquet

Right now you store:

retrieval_count
ranking_count
evaluation_count
production_count

but the reasoning generator later references:

production_evidence
retrieval_evidence

Those arrays must be persisted somewhere.

Otherwise reasoning generation cannot access them at runtime.

3. Add a title-chasing feature

The JD explicitly mentions it.

You identify it in philosophy but never score it.

Simple feature:

avg tenure per role
number of roles
promotion velocity

and apply a mild penalty.

Would I keep designing?

No.

At this point:

Stop architecture.
Start implementation.

You are well past the point where another design review will meaningfully improve results.

The biggest gains now come from:

inspecting real candidates
building the validation set
testing retrieval recall
running rankings
manually reviewing top-100 outputs

That's where competition-winning improvements usually come from.

Verdict: This is a strong final specification and is ready for implementation.
---

*End of specification. Version 2.0.0.*