## 4. Phase 0 — JD Intelligence

### 4.1 What this phase does

Runs **once offline**, no time constraint. Parses `JD_contract.yaml` into a structured config, generates **3 natural-language JD query strings** and encodes them via BGE-M3 into:
- 3 dense `.npy` vectors (for FAISS search)
- 1 learned-sparse `.npz` CSR query vector (for dot-product sparse retrieval)

Also builds the BM25 keyword list for lexical anchor retrieval.

**ColBERT multi-vectors are hard-disabled.** BGE-M3 ColBERT stores one 1024-dim vector per token per document. At 100K candidates with ~256 average tokens, that is ~100 GB of RAM. Hard-disable in all encode calls: `return_colbert_vecs=False`.

**GPU/CPU Auto-Detection:** The offline phase should exploit available GPU (Colab T4/A100 or AMD ROCm VM) automatically, falling back to multi-threaded CPU.

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = (device == "cuda")  # fp16 only safe on GPU
```

### 4.2 Structured JD config

Parse the JD into typed buckets and save as `artifacts/jd_config.json`:

```python
JD_CONFIG = {
    "must_have": [
        "production embeddings retrieval",
        "vector database hybrid search",
        "ranking system evaluation NDCG MRR MAP",
        "Python production code quality"
    ],
    "nice_to_have": [
        "LLM fine-tuning LoRA QLoRA",
        "learning to rank XGBoost LambdaMART",
        "HR-tech recruiting marketplace",
        "distributed inference optimization"
    ],
    "hard_disqualifiers": [
        "pure research no production",
        "consulting only no product company",
        "computer vision speech robotics no NLP",
        "LangChain only no pre-LLM experience",
        "no code written 18 months architecture only"
    ],
    "soft_negatives": [
        "title chaser switching every 1.5 years",
        "framework demo LangChain tutorial",
        "closed source only no external validation"
    ],
    "location_prefs": ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr"],
    "behavior_prefs": {
        "notice_period_days_max": 30,
        "open_to_work": True,
        "min_recruiter_response_rate": 0.50
    }
}
```

### 4.3 JD Query Strings — Three Variants

Use `BAAI/bge-m3` (dense + learned-sparse, ~570 MB). Load once and reuse across all three encodings.

**Why three query strings?**
- BGE-M3 is trained on natural sentence pairs. Syntactic structure guides token attention. A keyword dump (`"FAISS Pinecone BM25 NDCG"`) gives worse attention distribution than a grammatically coherent sentence.
- Each variant targets a different profile archetype that the JD explicitly says is a fit. One dense vector cannot capture both the IR vocabulary specialist and the RecSys engineer simultaneously.
- The third learned-sparse vector adds a complementary signal: it captures implicit vocabulary expansion (e.g., "vector store" bridging to "Milvus") that lexical BM25 misses.

**Variant 1: v1_skills — Auto-generated from YAML must_have + nice_to_have tokens**

Do not write this string by hand. Build it programmatically by flattening `JD_CONFIG` fields into human-readable sentences so it stays synchronized with `JD_contract.yaml`:

```python
def build_v1_skills_text(jd_config: dict) -> str:
    """
    Flatten JD_CONFIG into coherent natural-language sentences.
    This keeps the query synchronized with JD_contract.yaml without manual editing.
    """
    must = ". ".join(jd_config.get("must_have", []))
    nice = ". ".join(jd_config.get("nice_to_have", []))
    locations = ", ".join(jd_config.get("location_prefs", []))
    return (
        f"Senior AI Engineer with expertise in {must}. "
        f"Beneficial experience includes {nice}. "
        f"Located in or willing to relocate to {locations}. "
        "Strong Python engineering skills. Applied ML at product companies. "
        "Has shipped production ranking, search, or recommendation systems to real users at scale."
    )

JD_V1_SKILLS_TEXT = build_v1_skills_text(JD_CONFIG)
```

**Variant 2: HyDE Persona — RecSys / Marketplace Ranking Engineer**

HyDE (Hypothetical Document Embeddings, Gao et al. 2022) improves retrieval by mapping JD-space into candidate-resume-space. Instead of embedding the job description, we embed a synthetic candidate profile that would be a perfect fit. This closes the stylistic gap between an interrogative job description and descriptive resume text.

*Implementation note:* Standard HyDE uses an LLM per query call, adding 25–40% latency. Since Phase 0 runs offline once, we use three manually-crafted static persona blocks — same benefit, zero LLM runtime dependency.

```python
JD_HYDE_RECSYS_TEXT = """
I am a Senior ML Engineer with 7 years of experience building recommendation and ranking systems
at product companies. My most recent role involved designing and shipping a two-sided marketplace
matching engine that ranked 50M job candidates against 200K open roles daily. I implemented
collaborative filtering with matrix factorization, item and user embeddings, and a learning-to-rank
stage using XGBoost and LambdaMART. I have deep experience with candidate-job matching pipelines,
feed ranking systems, personalization engines, and real-time scoring infrastructure. I care deeply
about offline-online evaluation gap, run regular A/B tests, and measure ranking quality using
NDCG, MRR, and MAP. I have shipped from zero to production at a startup and understand the full
stack from embedding training to serving latency to business metric impact.
"""
```

**Variant 3: HyDE Persona — Evaluation / Metrics / Retrieval Depth Expert**

```python
JD_HYDE_EVAL_TEXT = """
I am a Senior AI Engineer specializing in information retrieval systems and evaluation
methodology with 6 years in applied ML at product companies. I built dense and sparse retrieval
pipelines using FAISS, Elasticsearch, and Qdrant, and implemented hybrid search combining BM25
lexical signals with bi-encoder dense vectors. I have designed rigorous evaluation frameworks
measuring NDCG@10, MRR, MAP, and Precision@K, and I understand the gap between offline benchmark
metrics and online A/B test outcomes. I have fine-tuned cross-encoders for reranking and trained
bi-encoders using contrastive learning on domain-specific datasets. I write production Python,
deploy inference services with low latency, and have worked on retrieval-augmented generation
pipelines with a strong preference for pre-LLM retrieval engineering foundations.
"""
```

### 4.4 BGE-M3 Encoding and Artifact Generation

**Hard rule: `return_colbert_vecs=False` on every encode call.**

```python
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import scipy.sparse
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = (device == "cuda")

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=use_fp16,
    device=device
)

queries = {
    "jd_v1_skills": JD_V1_SKILLS_TEXT,
    "jd_hyde_recsys": JD_HYDE_RECSYS_TEXT,
    "jd_hyde_eval": JD_HYDE_EVAL_TEXT,
}

sparse_dicts = []  # will hold the 3 query sparse dicts for CSR construction

for name, text in queries.items():
    output = model.encode(
        text,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False  # HARD DISABLED — see §3.3
    )
    np.save(f"artifacts/{name}_vector.npy", output["dense_vecs"])
    sparse_dicts.append(output["lexical_weights"])  # dict[int, float]

# Build a single consolidated sparse CSR query matrix (3 queries × vocab_size)
# Vocab size is dynamically inferred — never hardcoded.
vocab_size = max(max(d.keys()) for d in sparse_dicts) + 1

rows, cols, vals = [], [], []
for i, d in enumerate(sparse_dicts):
    for token_id, weight in d.items():
        rows.append(i)
        cols.append(token_id)
        vals.append(weight)

query_sparse_csr = scipy.sparse.csr_matrix(
    (vals, (rows, cols)),
    shape=(len(sparse_dicts), vocab_size)
)
scipy.sparse.save_npz("artifacts/jd_sparse_query.npz", query_sparse_csr)
print(f"Phase 0 complete. vocab_size={vocab_size}, query_sparse shape={query_sparse_csr.shape}")
```

**Why dynamic vocab_size?** The XLM-RoBERTa tokenizer used by BGE-M3 has ~250,002 token IDs (confirmed). But hardcoding any constant risks an index-out-of-bounds crash if the actual max observed token ID is higher. Dynamic inference (`max(max(d.keys()) for d in sparse_dicts) + 1`) costs nothing and eliminates this entire class of bug.

### 4.5 BM25 Keyword List

Used as the 5th signal in Phase 1d RRF. Kept as a separate, complementary signal to learned-sparse BGE-M3: BM25 is a literal term anchor for multi-word domain-specific phrases (e.g., `"schema drift"`, `"golden dataset"`, `"embedding drift"`) that BGE-M3's XLM-RoBERTa subword tokenizer fragments and weights poorly.

```python
JD_KEYWORDS = [
    "embeddings", "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
    "sentence-transformers", "BGE", "E5", "NDCG", "MRR", "MAP",
    "A/B test", "A/B testing", "vector search", "hybrid retrieval",
    "hybrid search", "BM25", "ranking", "reranking", "learning to rank",
    "LTR", "dense retrieval", "sparse retrieval", "retrieval system",
    "recommendation system", "search ranking", "relevance", "inverted index",
    "approximate nearest neighbor", "ANN", "evaluation framework",
    "offline evaluation", "online evaluation", "production ML", "inference",
    "semantic search", "bi-encoder", "cross-encoder", "reranker",
    "schema drift", "embedding drift", "golden dataset"
]
```

Save as `artifacts/jd_keywords.json`.

### 4.6 Step 1 Validation: test_sparse_pipeline.py

**This script must be written and executed before any other pipeline code is built.** It proves end-to-end mathematical integrity of the sparse storage and runtime dot-product path on 1,000 mock candidates.

The 9-step sequence:
1. **Load Model** — Initialize `BGEM3FlagModel` with GPU/CPU auto-detection and `return_colbert_vecs=False`.
2. **Mock Encode** — Pass 1,000 dummy candidate text strings through the encoder with `return_sparse=True`.
3. **Infer Vocab Size** — `vocab_size = max(max(d.keys()) for d in sparse_dicts) + 1`. Never hardcode.
4. **Build Candidate CSR** — Convert `list[dict[int, float]]` → `scipy.sparse.csr_matrix(shape=(1000, vocab_size))`.
5. **Save** — `scipy.sparse.save_npz("test_candidate_sparse.npz", candidate_csr)`.
6. **Load** — `candidate_csr = scipy.sparse.load_npz("test_candidate_sparse.npz")`.
7. **Encode JD Query** — Encode one sample JD text with `return_sparse=True`; get `query_dict: dict[int, float]`.
8. **Build Query CSR** — `scipy.sparse.csr_matrix(([...], ([0]*len(q), list(q.keys()))), shape=(1, vocab_size))` using the **same** `vocab_size` inferred in step 3.
9. **Dot Product + Validate** — `scores = candidate_csr.dot(query_csr.T).toarray().flatten()`. Assert: `len(scores) == 1000`, `np.isfinite(scores).all()`, `not np.isnan(scores).any()`.

> **Developer Note (Step 9):** The result of `.dot()` on a sparse matrix is itself sparse. You **must** chain `.toarray().flatten()` to get a dense NumPy float array. Failure to do this breaks all downstream RRF sorting.

---

