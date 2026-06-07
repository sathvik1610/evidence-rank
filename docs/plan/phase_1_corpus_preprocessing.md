## 5. Phase 1 — Corpus Preprocessing + Honeypot Detection

### 5.1 What this phase does

Streams all 100K candidates, serializes profile text, builds the FAISS vector index, flags honeypots and disqualifiers, and identifies ghost profiles. All outputs saved to `artifacts/`. No time constraint.

### 5.2 Profile text serialization

For each candidate, build a single text string for embedding:

```python
def build_profile_text(candidate):
    profile = candidate["profile"]
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_industry", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("company", ""))
        parts.append(role.get("description", ""))
    skill_names = " ".join(s["name"] for s in candidate.get("skills", []))
    parts.append(skill_names)
    return " ".join(p for p in parts if p).strip()
```

### 5.3 Corpus embedding — Dense FAISS index + Learned-Sparse CSR matrix

Both dense and learned-sparse outputs are generated in a **single encoding pass** over the 100K corpus. ColBERT is hard-disabled on every call.

**Batch size guidance:** GPU (Colab T4): batch_size=512 works well. CPU fallback: use batch_size=32–64 to avoid OOM.

```python
import faiss
import numpy as np
import scipy.sparse
import json
import torch
from FlagEmbedding import BGEM3FlagModel

device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = (device == "cuda")
batch_size = 512 if device == "cuda" else 32

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=use_fp16, device=device)

all_texts = []
all_ids = []
for candidate in stream_candidates("candidates.jsonl"):
    all_texts.append(build_profile_text(candidate))
    all_ids.append(candidate["candidate_id"])

# --- Encode in batches ---
all_dense = []
all_sparse_dicts = []  # list[dict[int, float]], one per candidate

for i in range(0, len(all_texts), batch_size):
    batch = all_texts[i:i + batch_size]
    output = model.encode(
        batch,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False  # HARD DISABLED
    )
    all_dense.append(output["dense_vecs"])
    all_sparse_dicts.extend(output["lexical_weights"])
    if (i // batch_size) % 20 == 0:
        print(f"  Encoded {i + len(batch):,} / {len(all_texts):,} candidates")

# --- Build FAISS index (dense) ---
embeddings = np.vstack(all_dense).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)  # L2-normalize for cosine via IP
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "artifacts/faiss_index.bin")
json.dump(all_ids, open("artifacts/candidate_ids.json", "w"))
print(f"FAISS index saved: {len(all_ids):,} candidates, dim={embeddings.shape[1]}")

# --- Build Candidate Sparse CSR matrix ---
# Vocab size is dynamically inferred — never hardcoded.
vocab_size = max(max(d.keys()) for d in all_sparse_dicts if d) + 1

rows, cols, vals = [], [], []
for i, d in enumerate(all_sparse_dicts):
    for token_id, weight in d.items():
        rows.append(i)
        cols.append(token_id)
        vals.append(float(weight))

candidate_sparse_csr = scipy.sparse.csr_matrix(
    (vals, (rows, cols)),
    shape=(len(all_sparse_dicts), vocab_size)
)
scipy.sparse.save_npz("artifacts/candidate_sparse.npz", candidate_sparse_csr)
print(f"Sparse CSR saved: shape={candidate_sparse_csr.shape}, nnz={candidate_sparse_csr.nnz:,}")
# Expected memory: ~80-100 MB for 100K candidates at ~100 avg non-zero tokens
# (vs 400 MB for dense, vs 25-100 GB for ColBERT)
```

### 5.4 BM25 index

```python
import pickle
from rank_bm25 import BM25Okapi

tokenized_corpus = [text.lower().split() for text in all_texts]
bm25 = BM25Okapi(tokenized_corpus)

with open("artifacts/bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)
with open("artifacts/candidate_texts.pkl", "wb") as f:
    pickle.dump(all_texts, f)
```

### 5.5 Phase 1f: Honeypot Audit Pass

This is a **completely separate pass** from the embedding pipeline. It reads raw JSON fields only — no ML, no text encoding. It runs after Phase 1 text encoding but before Phase 1d RRF scoring. It produces three outputs per candidate stored in `candidate_flags.parquet`.

**Why separate from embedding:** Honeypots are designed to have great keywords. BGE-M3 will score them highly. The contradictions live in structured fields (dates, durations, proficiency levels) that embedding models cannot see.

#### Tier 1 — `impossible_flag` (any ONE triggers it, multiplier = 0.01)

These are structural contradictions with no legitimate explanation:

```python
IMPOSSIBLE_TECH_RELEASES = {
    # Tool: (release_year, release_month) — only tools with precise, well-known dates
    "qdrant":     (2021, 6),
    "milvus":     (2019, 10),
    "pinecone":   (2019, 1),
    "langchain":  (2022, 10),
    "llamaindex": (2022, 11),
}
RELEASE_BUFFER_MONTHS = 12  # Safety margin against false positives

def check_impossible_flag(candidate) -> bool:
    career  = candidate.get("career_history", [])
    skills  = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    from datetime import date

    # Rule I-1: end_date before start_date for any role
    for role in career:
        sd = parse_date(role.get("start_date"))
        ed = parse_date(role.get("end_date"))
        if sd and ed and ed < sd:
            return True

    # Rule I-2: negative duration_months
    for role in career:
        if role.get("duration_months", 0) < 0:
            return True

    # Rule I-3: Technology claimed before it existed (+RELEASE_BUFFER_MONTHS safety)
    for skill in skills:
        name_lower = skill["name"].lower()
        for tech, (rel_year, rel_month) in IMPOSSIBLE_TECH_RELEASES.items():
            if tech in name_lower:
                release_date = date(rel_year, rel_month, 1)
                months_since_release = (date.today() - release_date).days / 30.436875
                claimed_months = skill.get("duration_months", 0)
                if claimed_months > months_since_release + RELEASE_BUFFER_MONTHS:
                    return True

    # Rule I-4: Total YoE mathematically impossible given earliest career start_date
    start_dates = [parse_date(r.get("start_date")) for r in career]
    start_dates = [d for d in start_dates if d]
    if start_dates:
        earliest = min(start_dates)
        max_possible_months = (date.today() - earliest).days / 30.436875
        claimed_months = profile.get("years_of_experience", 0) * 12
        if claimed_months > max_possible_months + 6:  # 6-month buffer for rounding
            return True

    return False
```

#### Tier 2 — `honeypot_score` (0.0–1.0 weighted soft sum)

Each contributor adds weight. Score is capped at 1.0.

```python
def compute_honeypot_score(candidate) -> float:
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    signals  = candidate.get("redrob_signals", {})
    profile  = candidate.get("profile", {})

    score = 0.0

    # S-1: Multiple simultaneous is_current=True roles with overlapping dates (weight 0.40)
    current_roles = [r for r in career if r.get("is_current")]
    if len(current_roles) >= 2:
        score += 0.40

    # S-2: YoE mismatch > 24 months buffer (weight 0.25)
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    claimed_months = profile.get("years_of_experience", 0) * 12
    if abs(total_career_months - claimed_months) > 24:
        score += 0.25

    # S-3: Expert skill with duration_months < 12 (weight 0.15 per occurrence, cap 0.30)
    expert_violations = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 999) < 12
    )
    score += min(0.30, expert_violations * 0.15)

    # S-4: All redrob_signals simultaneously maxed (weight 0.20)
    maxed = (
        signals.get("recruiter_response_rate", 0) >= 0.98 and
        signals.get("interview_completion_rate", 0) >= 0.98 and
        signals.get("offer_acceptance_rate", -1) >= 0.98 and
        signals.get("profile_completeness_score", 0) >= 99
    )
    if maxed:
        score += 0.20

    # S-5: github_activity_score=0 with 8+ years claimed applied ML (weight 0.15)
    if signals.get("github_activity_score", -1) == 0 and profile.get("years_of_experience", 0) >= 8:
        score += 0.15

    # S-6: All career descriptions statistically uniform in length (weight 0.10)
    desc_lengths = [len(r.get("description", "")) for r in career if r.get("description")]
    if len(desc_lengths) >= 3:
        mean_len = sum(desc_lengths) / len(desc_lengths)
        variance = sum((l - mean_len)**2 for l in desc_lengths) / len(desc_lengths)
        if variance < 100:  # All descriptions within ~10 chars of each other
            score += 0.10

    return min(score, 1.0)


def check_suspicious_flag(honeypot_score: float) -> bool:
    """Explicitly derived from honeypot_score. Stored in parquet alongside score."""
    return honeypot_score > 0.70
```

### 5.6 Ghost profile pre-filter

A ghost profile is one that is effectively unreachable regardless of fit. Pre-filtering removes them from retrieval entirely, improving precision without meaningful recall loss. Approximately 1–3% of the corpus will be flagged.

**Ghost criteria — ALL four conditions must be true:**
- `last_active_date` > 365 days before the reference date
- `recruiter_response_rate` < 0.05
- `open_to_work_flag` = False
- `applications_submitted_30d` = 0

```python
from datetime import date

def is_ghost(candidate, reference_date) -> bool:
    # Note: ghost pre-filtering is irreversible. The reference_date used here is locked
    # at preprocess time. If the evaluation payload contains newer dates, pre-filtered
    # candidates cannot be recovered.
    signals = candidate.get("redrob_signals", {})
    last_active_str = signals.get("last_active_date")
    if last_active_str is None:
        # Default to False (active). Missing dates could represent newly-imported profiles.
        return False
    days_inactive = (reference_date - date.fromisoformat(last_active_str)).days
    return (
        days_inactive > 365
        and signals.get("recruiter_response_rate", 1.0) < 0.05
        and not signals.get("open_to_work_flag", True)
        and signals.get("applications_submitted_30d", 1) == 0
    )
```

### 5.7 Disqualifier tagging

```python
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "l&t infotech", "niit technologies", "zensar", "mastech",
    "syntel", "kpit", "cyient", "birlasoft", "persistent systems"
}

def tag_disqualifiers(candidate) -> dict:
    career = candidate.get("career_history", [])
    titles_lower = [r.get("title", "").lower() for r in career]
    desc_text = " ".join(r.get("description", "").lower() for r in career)
    skills_lower = [s["name"].lower() for s in candidate.get("skills", [])]

    # consulting_only: ENTIRE career at consulting firms. If ANY non-consulting role is present, evaluates to False.
    total_months = sum(r.get("duration_months", 0) for r in career)
    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if any(firm in r.get("company", "").lower() for firm in CONSULTING_FIRMS)
        or r.get("industry", "").lower() in ("it services", "consulting", "outsourcing")
    )
    product_ratio = 1.0 - (consulting_months / total_months) if total_months > 0 else 0.0
    consulting_only = product_ratio == 0.0

    # research_only: only academic/research titles, no engineer/developer roles
    engineering_titles = {"engineer", "developer", "data scientist", "applied scientist", "architect", "lead", "head"}
    research_titles = {"researcher", "research scientist", "phd", "postdoc", "intern"}
    has_engineering = any(t in " ".join(titles_lower) for t in engineering_titles)
    has_only_research = not has_engineering and any(t in " ".join(titles_lower) for t in research_titles)
    research_only = has_only_research

    # wrong_domain: CV/speech/robotics without NLP/IR
    cv_speech_terms = {"computer vision", "opencv", "yolo", "object detection", "speech recognition", "tts", "asr", "robotics"}
    nlp_ir_terms = {"nlp", "retrieval", "ranking", "recommendation", "search", "embedding", "information retrieval"}
    has_cv_speech = any(t in desc_text or any(t in s for s in skills_lower) for t in cv_speech_terms)
    has_nlp_ir = any(t in desc_text or any(t in s for s in skills_lower) for t in nlp_ir_terms)
    wrong_domain = has_cv_speech and not has_nlp_ir

    return {
        "product_ratio": round(product_ratio, 4),
        "consulting_only": consulting_only,
        "research_only": research_only,
        "wrong_domain": wrong_domain
    }
```

### 5.8 Flags Parquet

Save `artifacts/candidate_flags.parquet` with one row per candidate:

| Column | Type | Notes |
|---|---|---|
| candidate_id | string | Primary key |
| impossible_flag | bool | True = Tier 1 hard contradiction. `final_score *= 0.01` |
| honeypot_score | float | 0.0–1.0 soft weighted sum of suspicious signals |
| suspicious_flag | bool | Derived: `honeypot_score > 0.70`. `final_score *= 0.01` |
| is_ghost | bool | True = pre-filtered dead profile |
| product_ratio | float | 0.0–1.0 time-weighted product company ratio |
| consulting_only | bool | Entire career at consulting firms |
| research_only | bool | Only academic/research roles |
| wrong_domain | bool | CV/speech/robotics, no NLP/IR |
| contradiction_skill_duration | int | Count of skills > career timeline + 48mo |
| contradiction_assessment | int | Count of expert skills with test score < 40 |

> [!IMPORTANT]
> `suspicious_flag` is computed during Phase 1f and stored explicitly in the parquet. It is **not** recomputed at runtime from `honeypot_score` — the parquet is the single source of truth. `impossible_flag` and `suspicious_flag` both result in `final_score *= 0.01` (not 0.0). Absolute zero is not used because a legitimate candidate triggering a false positive on the tech timeline check must still be recoverable by manual inspection.

---

