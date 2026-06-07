"""
constants.py — Single source of truth for all artifact paths, dataset paths, and model IDs.

Rules:
  - No logic, no imports, no computation — pure constants only.
  - Every artifact path in the codebase must reference this module.
  - Do NOT hardcode paths in preprocess.py, rank.py, or any src/ module.
  - All paths are relative to the repository root (where preprocess.py / rank.py live).
"""

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
CANDIDATES_JSONL = "Resources/candidates.jsonl"
SAMPLE_CANDIDATES_JSON = "Resources/sample_candidates.json"
SUBMISSION_TEMPLATE = "Resources/submission_metadata_template.yaml"
VALIDATE_SUBMISSION_SCRIPT = "Resources/validate_submission.py"

# ---------------------------------------------------------------------------
# Metadata / contracts
# ---------------------------------------------------------------------------
JD_CONTRACT_YAML = "metadata/JD_contract.yaml"
VALIDATION_SET_JSON = "metadata/validation_set.json"

# ---------------------------------------------------------------------------
# Artifacts directory
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"

# ---------------------------------------------------------------------------
# Phase 0 — JD Intelligence artifacts
# ---------------------------------------------------------------------------
JD_V1_SKILLS_NPY       = "artifacts/jd_v1_skills.npy"
JD_HYDE_RECSYS_NPY     = "artifacts/jd_hyde_recsys.npy"
JD_HYDE_EVAL_NPY       = "artifacts/jd_hyde_eval.npy"
JD_SPARSE_QUERIES_NPZ  = "artifacts/jd_sparse_queries.npz"
JD_KEYWORDS_JSON       = "artifacts/jd_keywords.json"
JD_CONFIG_JSON         = "artifacts/jd_config.json"

# ---------------------------------------------------------------------------
# Phase 1 — Corpus Preprocessing artifacts
# ---------------------------------------------------------------------------
FAISS_INDEX_BIN           = "artifacts/faiss_index.bin"
CANDIDATE_IDS_JSON        = "artifacts/candidate_ids.json"
CANDIDATE_SPARSE_NPZ      = "artifacts/candidate_sparse_matrix.npz"
CANDIDATE_TEXTS_PKL       = "artifacts/candidate_texts.pkl"
BM25_INDEX_PKL            = "artifacts/bm25_index.pkl"
CANDIDATE_FLAGS_PARQUET   = "artifacts/candidate_flags.parquet"
RUN_METADATA_JSON         = "artifacts/run_metadata.json"

# Phase 1d — RRF retrieval
RETRIEVAL_SCORES_PARQUET  = "artifacts/retrieval_scores.parquet"

# Phase 1c — Feature extraction (Bucket A/B/C, top 5000)
CANDIDATE_FEATURES_PARQUET = "artifacts/candidate_features.parquet"

# Phase 1e — Cross-encoder scores (top 500 offline)
CROSS_ENCODER_SCORES_PARQUET = "artifacts/cross_encoder_scores.parquet"

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SUBMISSION_CSV = "submission.csv"

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------
BGE_M3_MODEL_ID        = "BAAI/bge-m3"
BGE_RERANKER_MODEL_ID  = "BAAI/bge-reranker-v2-m3"

# ---------------------------------------------------------------------------
# Pipeline parameters (non-tunable structural constants — NOT in weights.yaml)
# These control algorithm topology, not score values. Do not move to weights.yaml.
# ---------------------------------------------------------------------------

# Maximum candidates precomputed in Phase 1d (RRF pool)
RRF_PRECOMPUTE_TOPK = 5000

# Batch sizes for BGE-M3 encoding (GPU vs CPU)
BATCH_SIZE_GPU = 512
BATCH_SIZE_CPU = 32

# Safety buffer for honeypot tech-release date check (months)
RELEASE_BUFFER_MONTHS = 12

# Ghost pre-filter: 4-condition AND gate (ALL must be true to mark ghost)
GHOST_DAYS_INACTIVE_THRESHOLD = 365
GHOST_RESPONSE_RATE_THRESHOLD = 0.05
GHOST_APPLICATIONS_THRESHOLD = 0       # applications_submitted_30d == 0

# Hard-rule buffers (Phase 1f)
SINGLE_ROLE_EXCEEDS_TIMELINE_BUFFER_MONTHS = 12  # Rule 2
CAREER_OVERLAP_RATIO_MAX = 1.5                   # Rule 3
YOE_IMPOSSIBLE_BUFFER_MONTHS = 6                 # Rule I-4

# Skill duration soft-flag buffer above chrono timeline (Phase 1f soft signals)
SKILL_DURATION_SOFT_FLAG_BUFFER_MONTHS = 48

# Cross-encoder: how many candidates to score offline
CE_PRECOMPUTE_TOPK = 5000  # Score all top-5000; runtime will slice top 500

# Honeypot score threshold for suspicious_flag
HONEYPOT_SUSPICIOUS_THRESHOLD = 0.70

# Known impossible technology release dates: {tech: (year, month)}
IMPOSSIBLE_TECH_RELEASES = {
    "qdrant":     (2021, 6),
    "milvus":     (2019, 10),
    "pinecone":   (2019, 1),
    "langchain":  (2022, 10),
    "llamaindex": (2022, 11),
}

# Consulting firms — used in product_ratio computation (Phase 1f soft signals + Phase 3)
CONSULTING_FIRMS = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "l&t infotech", "niit technologies", "zensar", "mastech",
    "syntel", "kpit", "cyient", "birlasoft", "persistent systems",
})

# Consulting industry labels (lowercase)
CONSULTING_INDUSTRIES = frozenset({"it services", "consulting", "outsourcing"})

# ---------------------------------------------------------------------------
# Runtime slice (rank.py slices RRF precompute to this many at runtime)
# ---------------------------------------------------------------------------
RUNTIME_RETRIEVAL_TOPK = 3000  # Sliced from RRF_PRECOMPUTE_TOPK at rank time
