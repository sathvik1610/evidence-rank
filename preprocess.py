"""
preprocess.py — Offline pipeline runner

Phase 0: JD Intelligence
Phase 1: Corpus Preprocessing (Embedding, FAISS, Sparse, BM25)
Phase 1f: Honeypot Detection (Candidate Flags)
Phase 1d: RRF Retrieval
Phase 1c: Feature Extraction
Phase 1e: Cross-Encoder Scoring
"""

import os
import sys
import json
import pickle
import argparse
import re
from datetime import date, datetime
import numpy as np
import pandas as pd
import scipy.sparse
import polars as pl
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

import constants
from src.features import extract_features, compute_product_ratio, _build_career_text
from src.weights import W
from src.jd_intelligence import build_feature_contract, build_jd_intelligence

def get_embedding_model():
    import torch
    from FlagEmbedding import BGEM3FlagModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")
    return BGEM3FlagModel(constants.BGE_M3_MODEL_ID, use_fp16=use_fp16, device=device)

def get_cross_encoder():
    import torch
    from FlagEmbedding import FlagReranker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")
    return FlagReranker(constants.BGE_RERANKER_MODEL_ID, use_fp16=use_fp16)

# ---------------------------------------------------------------------------
# Date and Text Helpers
# ---------------------------------------------------------------------------

def parse_date(d_str: str) -> date:
    if not d_str:
        return None
    try:
        if len(d_str) == 7:
            return datetime.strptime(d_str, "%Y-%m").date()
        elif len(d_str) == 10:
            return datetime.strptime(d_str, "%Y-%m-%d").date()
        elif len(d_str) == 4:
            return datetime.strptime(d_str, "%Y").date()
    except Exception:
        pass
    return None

def build_profile_text(candidate: dict) -> str:
    """Serialize candidate dict into a single string for embedding/BM25."""
    profile = candidate.get("profile", {})
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
    skill_names = " ".join(s.get("name", "") for s in candidate.get("skills", []))
    parts.append(skill_names)
    return " ".join(str(p) for p in parts if p).strip()

def normalize_text(text: str) -> str:
    """Used for BM25 to ensure tokenization alignment."""
    text = text.lower()
    text = re.sub(r"[/\\_-]+", " ", text)
    text = re.sub(r"[^a-z0-9@.+#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Phase 0: JD Intelligence
# ---------------------------------------------------------------------------

def run_phase_0(model):
    print("\n--- Phase 0: JD Intelligence ---")
    os.makedirs(constants.ARTIFACTS_DIR, exist_ok=True)

    jd_intel = build_jd_intelligence(constants.JD_CONTRACT_YAML, constants.JD_TEXT)
    
    with open(constants.JD_CONFIG_JSON, "w") as f:
        json.dump(jd_intel["config"], f, indent=2)
    with open(constants.JD_KEYWORDS_JSON, "w") as f:
        json.dump(jd_intel["keywords"], f, indent=2)
        
    queries = jd_intel["queries"]
    
    sparse_dicts = []
    
    for name, text in queries.items():
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                output = model.encode(
                    text,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False
                )
            finally:
                sys.stderr = old_stderr
        # BGE-M3 model.encode returns a dict or list depending on input. For single string, it returns single item lists.
        # But wait, it might return a dict like {'dense_vecs': ndarray, 'lexical_weights': dict}.
        dense_vec = output["dense_vecs"]
        if dense_vec.ndim == 1:
            dense_vec = dense_vec.reshape(1, -1)
        
        path = getattr(constants, f"JD_{name.upper()}_NPY")
        np.save(path, dense_vec)
        
        lex_weights = output["lexical_weights"]
        if isinstance(lex_weights, list):
            lex_weights = lex_weights[0]
        sparse_dicts.append(lex_weights)

    non_empty = [d for d in sparse_dicts if d]
    vocab_size = max(max(int(k) for k in d.keys()) for d in non_empty) + 1 if non_empty else 1
    
    rows, cols, vals = [], [], []
    for i, d in enumerate(sparse_dicts):
        for token_id, weight in d.items():
            rows.append(i)
            cols.append(int(token_id))
            vals.append(float(weight))
            
    query_sparse_csr = scipy.sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(sparse_dicts), vocab_size)
    )
    scipy.sparse.save_npz(constants.JD_SPARSE_QUERIES_NPZ, query_sparse_csr)
    print(f"Saved JD artifacts. Sparse vocab size: {vocab_size}")


# ---------------------------------------------------------------------------
# Phase 1: Corpus Preprocessing
# ---------------------------------------------------------------------------

def run_phase_1(model, candidates, batch_size):
    print(f"\n--- Phase 1: Corpus Preprocessing (Embedding) ---")
    import faiss
    from rank_bm25 import BM25Okapi
    
    all_texts = []
    all_ids = []
    
    print("Building texts...")
    for cand in candidates:
        all_texts.append(build_profile_text(cand))
        all_ids.append(cand.get("candidate_id", "UNKNOWN"))
        
    print("Encoding dense and sparse vectors...")
    all_dense = []
    all_sparse_dicts = []
    
    for i in tqdm(range(0, len(all_texts), batch_size)):
        batch = all_texts[i:i + batch_size]
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                output = model.encode(
                    batch,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False
                )
            finally:
                sys.stderr = old_stderr
        all_dense.append(output["dense_vecs"])
        all_sparse_dicts.extend(output["lexical_weights"])

    embeddings = np.vstack(all_dense).astype(np.float32)
    # L2 normalize using safe numpy to avoid missing FAISS attributes
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1e-10, norms)
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, constants.FAISS_INDEX_BIN)
    
    with open(constants.CANDIDATE_IDS_JSON, "w") as f:
        json.dump(all_ids, f)
        
    print(f"FAISS index saved: {len(all_ids):,} candidates, dim={embeddings.shape[1]}")
    
    # Sparse CSR
    non_empty_all = [d for d in all_sparse_dicts if d]
    vocab_size = max(max(int(k) for k in d.keys()) for d in non_empty_all) + 1 if non_empty_all else 1
    rows, cols, vals = [], [], []
    for i, d in enumerate(all_sparse_dicts):
        for token_id, weight in d.items():
            rows.append(i)
            cols.append(int(token_id))
            vals.append(float(weight))
            
    candidate_sparse_csr = scipy.sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(all_sparse_dicts), vocab_size)
    )
    scipy.sparse.save_npz(constants.CANDIDATE_SPARSE_NPZ, candidate_sparse_csr)
    print(f"Sparse CSR saved: shape={candidate_sparse_csr.shape}")

    # BM25
    print("Building BM25 Index...")
    tokenized_corpus = [normalize_text(text).split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(constants.BM25_INDEX_PKL, "wb") as f:
        pickle.dump(bm25, f)
    with open(constants.CANDIDATE_TEXTS_PKL, "wb") as f:
        pickle.dump(all_texts, f)


# ---------------------------------------------------------------------------
# Phase 1f: Honeypot Detection
# ---------------------------------------------------------------------------

def _check_impossible_flag(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    
    # I-1: end before start
    for role in career:
        sd = parse_date(role.get("start_date"))
        ed = parse_date(role.get("end_date"))
        if sd and ed and ed < sd:
            return True
            
    # I-2: negative duration
    for role in career:
        if role.get("duration_months", 0) < 0:
            return True
            
    # I-3: Technology claimed before existed
    for skill in skills:
        name_lower = skill.get("name", "").lower()
        for tech, (rel_year, rel_month) in constants.IMPOSSIBLE_TECH_RELEASES.items():
            if tech in name_lower:
                release_date = date(rel_year, rel_month, 1)
                months_since_release = (date.today() - release_date).days / 30.436875
                claimed_months = skill.get("duration_months", 0)
                if claimed_months > months_since_release + constants.RELEASE_BUFFER_MONTHS:
                    return True
                    
    # I-4: Total YoE impossible
    start_dates = [parse_date(r.get("start_date")) for r in career]
    start_dates = [d for d in start_dates if d]
    if start_dates:
        earliest = min(start_dates)
        max_possible_months = (date.today() - earliest).days / 30.436875
        claimed_months = profile.get("years_of_experience", 0) * 12
        if claimed_months > max_possible_months + constants.YOE_IMPOSSIBLE_BUFFER_MONTHS:
            return True
            
    # I-5: Skill duration wildly exceeds claimed YoE.
    # Skill-duration fields are noisy and overlapping, so a small overshoot should
    # become a contradiction penalty, not a hard kill. Keep this rule for absurd
    # claims only.
    yoe_months = profile.get("years_of_experience", 0) * 12
    if any(
        (s.get("duration_months") or 0) > yoe_months + constants.SKILL_DURATION_SOFT_FLAG_BUFFER_MONTHS
        for s in skills
    ):
        return True
        
    # I-6: Expert/Advanced skill failed basic assessment (from validation_set rules)
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if any(
        s.get("proficiency") in ("expert", "advanced") 
        and s.get("name") in assessment_scores 
        and assessment_scores[s.get("name")] < 40 
        for s in skills
    ):
        return True
            
    return False

def _compute_honeypot_score(candidate: dict) -> float:
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    
    score = 0.0
    
    # S-1
    current_roles = [r for r in career if r.get("is_current")]
    if len(current_roles) >= 2:
        score += W["honeypot.s1_multi_current_roles"]
        
    # S-2
    total_career_months = sum(r.get("duration_months", 0) or 0 for r in career)
    claimed_months = profile.get("years_of_experience", 0) * 12
    if abs(total_career_months - claimed_months) > 24:
        score += W["honeypot.s2_yoe_mismatch"]
        
    # S-3
    expert_violations = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 999) < 12
    )
    score += min(0.30, expert_violations * W["honeypot.s3_expert_short_duration"])
    
    # S-4
    maxed = (
        signals.get("recruiter_response_rate", 0) >= 0.98 and
        signals.get("interview_completion_rate", 0) >= 0.98 and
        signals.get("offer_acceptance_rate", -1) >= 0.98 and
        signals.get("profile_completeness_score", 0) >= 99
    )
    if maxed:
        score += W["honeypot.s4_maxed_signals"]
        
    # S-5
    if signals.get("github_activity_score", -1) == 0 and profile.get("years_of_experience", 0) >= 8:
        score += W["honeypot.s5_github_zero_senior"]
        
    # S-6
    desc_lengths = [len(r.get("description", "")) for r in career if r.get("description")]
    if len(desc_lengths) >= 3:
        mean_len = sum(desc_lengths) / len(desc_lengths)
        variance = sum((l - mean_len)**2 for l in desc_lengths) / len(desc_lengths)
        if variance < 100:
            score += W["honeypot.s6_uniform_descriptions"]
            
    return min(score, 1.0)


def _is_ghost(candidate: dict, reference_date: date) -> bool:
    signals = candidate.get("redrob_signals", {})
    last_active_str = signals.get("last_active_date")
    if not last_active_str:
        return False
    try:
        last_active = date.fromisoformat(last_active_str)
        days_inactive = (reference_date - last_active).days
    except ValueError:
        return False

    return (
        days_inactive > constants.GHOST_DAYS_INACTIVE_THRESHOLD
        and signals.get("recruiter_response_rate", 1.0) < constants.GHOST_RESPONSE_RATE_THRESHOLD
        and not signals.get("open_to_work_flag", True)
        and signals.get("applications_submitted_30d", 1) == constants.GHOST_APPLICATIONS_THRESHOLD
    )


def run_phase_1f_honeypots(candidates):
    print("\n--- Phase 1f: Honeypot Detection ---")
    feature_contract = build_feature_contract(constants.JD_CONTRACT_YAML)
    
    # Find reference date (max last_active_date)
    max_date = date(1970, 1, 1)
    for c in candidates:
        d_str = c.get("redrob_signals", {}).get("last_active_date")
        if d_str:
            try:
                d = date.fromisoformat(d_str)
                if d > max_date:
                    max_date = d
            except ValueError:
                pass
    
    print(f"Reference date for ghost detection: {max_date}")

    records = []
    for c in tqdm(candidates, desc="Flagging candidates"):
        cid = c.get("candidate_id")
        
        # Ghost
        is_ghost = _is_ghost(c, max_date)
        
        # Honeypot
        impossible = _check_impossible_flag(c)
        hp_score = _compute_honeypot_score(c)
        suspicious = hp_score > W["honeypot.suspicious_threshold"]
        
        # Disqualifiers
        career = c.get("career_history", [])
        titles_lower = " ".join([r.get("title", "").lower() for r in career])
        desc_text = " ".join(r.get("description", "").lower() for r in career)
        skills_lower = [s.get("name", "").lower() for s in c.get("skills", [])]
        
        product_ratio = compute_product_ratio(c)
        consulting_only = (product_ratio == 0.0)
        
        engineering_titles = {"engineer", "developer", "data scientist", "applied scientist", "architect", "lead", "head"}
        research_titles = set(feature_contract["research_title_terms"]) | {"researcher", "phd", "postdoc", "intern"}
        has_eng = any(t in titles_lower for t in engineering_titles)
        has_res = any(t in titles_lower for t in research_titles)
        research_only = has_res and not has_eng
        
        cv_terms = set(feature_contract["wrong_domain_terms"])
        nlp_terms = set(feature_contract["wrong_domain_escape_terms"])
        has_cv = any(t in desc_text or any(t in s for s in skills_lower) for t in cv_terms)
        has_nlp = any(t in desc_text or any(t in s for s in skills_lower) for t in nlp_terms)
        wrong_domain = has_cv and not has_nlp

        # BUG 2 FIX: Compute contradiction counts for consistency_score in behavioral.py
        # contradiction_skill_duration: skills claimed longer than career timeline + 48mo buffer
        total_career_months = sum(r.get("duration_months", 0) or 0 for r in career)
        contradiction_skill_duration = sum(
            1 for s in c.get("skills", [])
            if (s.get("duration_months") or 0) > total_career_months + 48
        )

        # contradiction_assessment: expert skills where assessment score < 40
        assessment_scores = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
        contradiction_assessment = sum(
            1 for s in c.get("skills", [])
            if s.get("proficiency") == "expert"
            and s.get("name") in assessment_scores
            and assessment_scores[s["name"]] < 40
        )

        records.append({
            "candidate_id": cid,
            "impossible_flag": impossible,
            "honeypot_score": round(hp_score, 4),
            "suspicious_flag": suspicious,
            "is_ghost": is_ghost,
            "product_ratio": round(product_ratio, 4),
            "consulting_only": consulting_only,
            "research_only": research_only,
            "wrong_domain": wrong_domain,
            "contradiction_skill_duration": contradiction_skill_duration,
            "contradiction_assessment": contradiction_assessment,
        })
        
    df = pd.DataFrame(records)
    df.to_parquet(constants.CANDIDATE_FLAGS_PARQUET)
    print(f"Flags saved to {constants.CANDIDATE_FLAGS_PARQUET}")
    # Return max_date so main() can persist it as reference_date
    return max_date


# ---------------------------------------------------------------------------
# Phase 1d: RRF Retrieval
# ---------------------------------------------------------------------------

def run_phase_1d_rrf():
    print("\n--- Phase 1d: RRF Retrieval ---")
    import faiss
    
    # Load indices
    index = faiss.read_index(constants.FAISS_INDEX_BIN)
    with open(constants.CANDIDATE_IDS_JSON, "r") as f:
        all_ids = json.load(f)
        
    jd_v1 = np.load(constants.JD_V1_SKILLS_NPY).astype(np.float32)
    jd_recsys = np.load(constants.JD_HYDE_RECSYS_NPY).astype(np.float32)
    jd_eval = np.load(constants.JD_HYDE_EVAL_NPY).astype(np.float32)
    
    # 1, 2, 3: Dense
    k_search = min(len(all_ids), constants.RRF_PRECOMPUTE_TOPK)
    _, idx1 = index.search(jd_v1, k=k_search)
    _, idx2 = index.search(jd_recsys, k=k_search)
    _, idx3 = index.search(jd_eval, k=k_search)
    dense_ids_v1 = [all_ids[i] for i in idx1[0]]
    dense_ids_recsys = [all_ids[i] for i in idx2[0]]
    dense_ids_eval = [all_ids[i] for i in idx3[0]]
    
    # 4: Sparse
    candidate_csr = scipy.sparse.load_npz(constants.CANDIDATE_SPARSE_NPZ)
    jd_sparse = scipy.sparse.load_npz(constants.JD_SPARSE_QUERIES_NPZ)
    
    vocab_size = candidate_csr.shape[1]
    if jd_sparse.shape[1] < vocab_size:
        jd_sparse = scipy.sparse.hstack([
            jd_sparse,
            scipy.sparse.csr_matrix((jd_sparse.shape[0], vocab_size - jd_sparse.shape[1]))
        ])
    elif jd_sparse.shape[1] > vocab_size:
        jd_sparse = jd_sparse[:, :vocab_size]
        
    # Keep RRF as a 5-channel fusion, but let the learned-sparse channel see all
    # YAML-derived query rows instead of only the first skills row.
    sparse_scores = candidate_csr.dot(jd_sparse.T).toarray().max(axis=1)
    top_sparse_idx = np.argsort(sparse_scores)[::-1][:k_search]
    sparse_ids = [all_ids[i] for i in top_sparse_idx]
    
    # 5: BM25
    with open(constants.BM25_INDEX_PKL, "rb") as f:
        bm25 = pickle.load(f)
    with open(constants.JD_KEYWORDS_JSON, "r") as f:
        keywords = json.load(f)
        
    query_tokens = normalize_text(" ".join(keywords)).split()
    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:k_search]
    sparse_ids_bm25 = [all_ids[i] for i in top_bm25_idx]
    
    # RRF fusion
    def rrf(ranked_lists, k=60):
        scores = {}
        for ranked in ranked_lists:
            for rank_idx, cand_id in enumerate(ranked):
                scores[cand_id] = scores.get(cand_id, 0.0) + 1.0 / (k + rank_idx + 1)
        return scores
        
    rrf_scores = rrf([
        dense_ids_v1, dense_ids_recsys, dense_ids_eval, sparse_ids, sparse_ids_bm25
    ])
    
    retrieved = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:constants.RRF_PRECOMPUTE_TOPK]
    
    df = pd.DataFrame(retrieved, columns=["candidate_id", "rrf_score"])
    df.to_parquet(constants.RETRIEVAL_SCORES_PARQUET)
    print(f"RRF retrieval saved to {constants.RETRIEVAL_SCORES_PARQUET} ({len(df)} candidates)")


# ---------------------------------------------------------------------------
# Phase 1c: Feature Extraction
# ---------------------------------------------------------------------------

def run_phase_1c_features(candidates, is_skip_embed=False):
    print("\n--- Phase 1c: Feature Extraction ---")
    
    # Read flags
    flags_df = pl.read_parquet(constants.CANDIDATE_FLAGS_PARQUET)
    flags_dict = {row["candidate_id"]: row for row in flags_df.to_dicts()}
    
    # Determine which candidates to extract
    if is_skip_embed:
        target_ids = {c["candidate_id"] for c in candidates}
    else:
        retrieval_df = pl.read_parquet(constants.RETRIEVAL_SCORES_PARQUET)
        target_ids = set(retrieval_df["candidate_id"].to_list())
        
    extracted = []
    for c in tqdm(candidates, desc="Extracting features"):
        cid = c.get("candidate_id")
        if cid in target_ids:
            cf = flags_dict.get(cid, {})
            # features.py expects dict with Phase 1f output
            f = extract_features(c, cf)
            extracted.append(f)
            
    df = pd.DataFrame(extracted)
    df.to_parquet(constants.CANDIDATE_FEATURES_PARQUET)
    print(f"Feature extraction saved to {constants.CANDIDATE_FEATURES_PARQUET} ({len(df)} candidates)")


# ---------------------------------------------------------------------------
# Phase 1e: Cross-Encoder
# ---------------------------------------------------------------------------

def run_phase_1e_cross_encoder(candidates, model):
    print("\n--- Phase 1e: Cross-Encoder ---")
    retrieval_df = pl.read_parquet(constants.RETRIEVAL_SCORES_PARQUET)
    top_ids = retrieval_df["candidate_id"].head(constants.CE_PRECOMPUTE_TOPK).to_list()
    top_set = set(top_ids)
    
    jd_v1 = build_jd_intelligence(constants.JD_CONTRACT_YAML, constants.JD_TEXT)["cross_encoder_query"]
    
    pairs = []
    pair_cids = []
    for c in candidates:
        if c.get("candidate_id") in top_set:
            text = build_profile_text(c)
            pairs.append([jd_v1, text])
            pair_cids.append(c.get("candidate_id"))
            
    print(f"Scoring {len(pairs)} candidates with cross-encoder...")
    # Process in chunks to avoid OOM
    batch_size = 32
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size)):
        batch = pairs[i:i + batch_size]
        batch_scores = model.compute_score(batch)
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
        
    df = pd.DataFrame({"candidate_id": pair_cids, "ce_score": scores})
    df.to_parquet(constants.CROSS_ENCODER_SCORES_PARQUET)
    print(f"Cross-encoder scores saved to {constants.CROSS_ENCODER_SCORES_PARQUET}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline Corpus Preprocessing Pipeline")
    parser.add_argument("--candidates", type=str, default=constants.CANDIDATES_JSONL, help="Path to candidates.jsonl")
    parser.add_argument("--skip-embed", action="store_true", help="Skip heavy embedding phases for local testing (runs honeypot & features only)")
    parser.add_argument("--sample", action="store_true", help="Run on a small sample of candidates (usually 50)")
    args = parser.parse_args()
    
    input_file = args.candidates
    if args.sample:
        input_file = constants.SAMPLE_CANDIDATES_JSON
        
    print(f"Loading candidates from {input_file}...")
    
    candidates = []
    if input_file.endswith(".jsonl"):
        with open(input_file, "r") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        with open(input_file, "r") as f:
            candidates = json.load(f)
            
    print(f"Loaded {len(candidates)} candidates.")
    
    if not args.skip_embed:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = constants.BATCH_SIZE_GPU if device == "cuda" else constants.BATCH_SIZE_CPU
        print(f"Detected device: {device}. Using batch_size={batch_size}")
        
        embed_model = get_embedding_model()
        run_phase_0(embed_model)
        run_phase_1(embed_model, candidates, batch_size)
    else:
        print("\nSkipping Phase 0 and Phase 1 Embedding (--skip-embed)")
        
    # Phase 1f (Honeypot) - Runs always
    reference_date = run_phase_1f_honeypots(candidates)
    
    if not args.skip_embed:
        # Phase 1d (RRF) - Requires embeddings
        run_phase_1d_rrf()
        
    # Phase 1c (Features) - Works with or without embeddings
    run_phase_1c_features(candidates, is_skip_embed=args.skip_embed)
    
    if not args.skip_embed:
        # Phase 1e (Cross-Encoder) - Requires GPU + RRF output
        ce_model = get_cross_encoder()
        run_phase_1e_cross_encoder(candidates, ce_model)
        
    # Save run metadata
    os.makedirs(constants.ARTIFACTS_DIR, exist_ok=True)
    with open(constants.RUN_METADATA_JSON, "w") as f:
        json.dump({
            "reference_date": reference_date.isoformat(),  # GAP 3 FIX: max(last_active_date)
            "run_time": datetime.utcnow().isoformat() + "Z",
            "candidate_count": len(candidates),
            "skip_embed": args.skip_embed
        }, f, indent=2)
        
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
