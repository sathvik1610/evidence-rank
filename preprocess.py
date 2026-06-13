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
import math
import shutil
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

EVAL_ADJACENT_RECALL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bengagement signals?\b",
        r"\bimplicit[-\s]+feedback\b",
        r"\bclick signals?\b",
        r"\bconversion signals?\b",
        r"\bfeedback loops?\b",
        r"\bquality regression\b",
        r"\bretrieval[-\s]+quality regression\b",
        r"\bgradient[-\s]+boosted re[-\s]+ranking\b",
    )
]

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
    return FlagReranker(constants.CE_RERANKER_MODEL, use_fp16=use_fp16)

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
            
    # I-3 reserved: Do not hard-kill from external technology release dates.
    # The reproducible ranking should rely on contradictions visible in the
    # released candidate JSONL, not outside product-history knowledge.

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

    # I-6b: Official honeypot-style trap from submission_spec.txt:
    # many expert claims with zero usage duration. The corpus audit showed
    # isolated expert-zero skills exist, but candidate-level piles do not,
    # so this stays narrow and avoids punishing ordinary synthetic noise.
    expert_zero_count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) <= 0
    )
    if expert_zero_count >= 8:
        return True

    # I-7: Long role descriptions copied verbatim across three or more
    # different employers. This is an adversarial synthetic-profile pattern,
    # not normal resume summarization.
    if _has_repeated_role_descriptions(candidate):
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


def _normalize_role_description(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _has_repeated_role_descriptions(candidate: dict) -> bool:
    """Detect copied long descriptions across multiple companies."""
    career = candidate.get("career_history", [])
    desc_to_companies: dict[str, set[str]] = {}
    for role in career:
        desc = _normalize_role_description(role.get("description", ""))
        if len(desc) < 160:
            continue
        desc_to_companies.setdefault(desc, set()).add((role.get("company") or "").strip().lower())

    return any(len(companies) >= 3 for companies in desc_to_companies.values())


def _target_skill_duration_contradictions(candidate: dict) -> tuple[int, float]:
    """Count target-domain skill duration claims that exceed claimed YoE."""
    yoe_months = (candidate.get("profile", {}).get("years_of_experience", 0) or 0) * 12
    if yoe_months <= 0:
        return 0, 0.0

    contradiction_count = 0
    max_overclaim = 0.0
    target_terms = constants.TARGET_SKILL_DURATION_TERMS
    target_proficiencies = {"expert", "advanced"}

    for skill in candidate.get("skills", []):
        name = (skill.get("name") or "").lower()
        if not name or not any(term in name for term in target_terms):
            continue
        if (skill.get("proficiency") or "").lower() not in target_proficiencies:
            continue

        duration = skill.get("duration_months") or 0
        overclaim = duration - (yoe_months + constants.TARGET_SKILL_DURATION_BUFFER_MONTHS)
        if overclaim > 0:
            contradiction_count += 1
            max_overclaim = max(max_overclaim, overclaim)

    return contradiction_count, round(float(max_overclaim), 2)


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
        target_contradictions, max_target_overclaim = _target_skill_duration_contradictions(c)

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
            "target_skill_duration_contradiction": target_contradictions,
            "max_target_skill_overclaim_months": max_target_overclaim,
        })
        
    df = pd.DataFrame(records)
    df.to_parquet(constants.CANDIDATE_FLAGS_PARQUET)
    print(f"Flags saved to {constants.CANDIDATE_FLAGS_PARQUET}")
    # Return max_date so main() can persist it as reference_date
    return max_date


# ---------------------------------------------------------------------------
# Phase 1d: RRF Retrieval
# ---------------------------------------------------------------------------

def _compile_recall_patterns(feature_contract: dict) -> dict[str, list[re.Pattern]]:
    """Compile field-aware exact recall patterns from the JD contract."""
    pattern_groups = {
        "primary": (
            feature_contract["retrieval_patterns"]
            + feature_contract["ranking_patterns"]
            + feature_contract["system_semantics_patterns"]
        ),
        "vector": feature_contract["target_skills"]["vector_db_hybrid"],
        "eval": feature_contract["target_skills"]["eval_framework"],
        "python": feature_contract["target_skills"]["python_coding"],
        "production": feature_contract["production_patterns"],
    }
    return {
        name: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for name, patterns in pattern_groups.items()
    }


def _count_pattern_hits(text: str, patterns: list[re.Pattern]) -> int:
    if not text:
        return 0
    return sum(1 for pattern in patterns if pattern.search(text))


def _score_exact_recall_candidate(candidate: dict, compiled_patterns: dict[str, list[re.Pattern]]) -> float:
    """
    Field-aware lexical rescue score.

    This lane is deliberately recall-oriented but not a generic keyword counter:
    it requires at least one primary retrieval/ranking/recsys signal, then rewards
    career-history evidence more than skills-only claims.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    current_title = str(profile.get("current_title") or "")
    headline = str(profile.get("headline") or "")
    summary = str(profile.get("summary") or "")
    title_text = " ".join(
        [current_title, headline]
        + [str(role.get("title") or "") for role in career]
    )
    desc_text = " ".join(str(role.get("description") or "") for role in career)
    summary_text = " ".join([summary, str(profile.get("current_industry") or "")])
    skills_text = " ".join(str(skill.get("name") or "") for skill in skills)

    primary_title = _count_pattern_hits(title_text, compiled_patterns["primary"])
    primary_desc = _count_pattern_hits(desc_text, compiled_patterns["primary"])
    primary_summary = _count_pattern_hits(summary_text, compiled_patterns["primary"])
    primary_skills = _count_pattern_hits(skills_text, compiled_patterns["primary"])
    primary_total = primary_title + primary_desc + primary_summary + primary_skills
    if primary_total == 0:
        return 0.0

    vector_desc = _count_pattern_hits(desc_text, compiled_patterns["vector"])
    vector_skills = _count_pattern_hits(skills_text, compiled_patterns["vector"])
    eval_desc = _count_pattern_hits(desc_text, compiled_patterns["eval"])
    eval_skills = _count_pattern_hits(skills_text, compiled_patterns["eval"])
    eval_adjacent_desc = _count_pattern_hits(desc_text, EVAL_ADJACENT_RECALL_PATTERNS)
    python_hits = _count_pattern_hits(desc_text + " " + skills_text, compiled_patterns["python"])
    production_hits = _count_pattern_hits(desc_text, compiled_patterns["production"])
    production_core = production_hits > 0 and (primary_desc > 0 or vector_desc > 0)
    production_vector = production_hits > 0 and vector_desc > 0
    eval_career = eval_desc > 0 or (eval_adjacent_desc > 0 and production_core)

    # Cap each component so verbose profiles do not dominate by repetition.
    score = (
        7.0 * min(primary_desc, 4)
        + 3.0 * min(primary_title, 3)
        + 2.0 * min(primary_summary, 2)
        + 0.75 * min(primary_skills, 5)
        + 4.0 * min(vector_desc, 3)
        + 0.50 * min(vector_skills, 4)
        + 4.0 * min(eval_desc, 3)
        + 2.0 * min(eval_adjacent_desc, 2)
        + 0.25 * min(eval_skills, 3)
        + 0.75 * min(python_hits, 3)
        + 3.0 * min(production_hits, 4)
        + (6.0 if production_core else 0.0)
        + (4.0 if production_vector else 0.0)
        + (4.0 if eval_career else 0.0)
    )

    # Require either career-description evidence or multiple corroborating fields.
    # This suppresses pure skill-list stuffing while still rescuing sparse profiles.
    if primary_desc == 0 and primary_title == 0 and primary_skills < 2:
        return 0.0
    if primary_desc == 0 and production_hits == 0 and vector_desc == 0 and eval_desc == 0:
        score *= 0.55

    yoe = profile.get("years_of_experience")
    if isinstance(yoe, (int, float)):
        if 4.0 <= yoe <= 12.0:
            score *= 1.05
        elif yoe < 3.0 or yoe > 15.0:
            score *= 0.90

    return float(score)


def build_exact_recall_ranked_list(candidates: list[dict], ghost_ids: set[str] | None = None) -> list[str]:
    """
    Build a high-recall exact/regex candidate list over the whole corpus.

    This uses only raw candidate JSON and the JD contract, so it can run during
    --skip-embed to widen the feature pool without recomputing BGE embeddings.
    """
    ghost_ids = ghost_ids or set()
    feature_contract = build_feature_contract(constants.JD_CONTRACT_YAML)
    compiled_patterns = _compile_recall_patterns(feature_contract)

    scored: list[tuple[str, float]] = []
    for candidate in candidates:
        cid = candidate.get("candidate_id")
        if not cid or cid in ghost_ids:
            continue
        score = _score_exact_recall_candidate(candidate, compiled_patterns)
        if score > 0.0 and math.isfinite(score):
            scored.append((cid, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [cid for cid, _ in scored[:constants.EXACT_RECALL_TOPK]]


def _rrf(ranked_lists, k=int(W["retrieval.rrf_k"])):
    scores = {}
    for ranked in ranked_lists:
        for rank_idx, cand_id in enumerate(ranked):
            scores[cand_id] = scores.get(cand_id, 0.0) + 1.0 / (k + rank_idx + 1)
    return scores


def run_phase_1d_rrf(candidates=None):
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
    base_retrieved = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:constants.RRF_PRECOMPUTE_TOPK]
    pd.DataFrame(base_retrieved, columns=["candidate_id", "rrf_score"]).to_parquet(
        constants.BASE_RETRIEVAL_SCORES_PARQUET
    )

    if candidates is not None:
        ghost_ids = set()
        if os.path.exists(constants.CANDIDATE_FLAGS_PARQUET):
            flags_df = pl.read_parquet(constants.CANDIDATE_FLAGS_PARQUET)
            if "is_ghost" in flags_df.columns:
                ghost_ids = set(
                    flags_df.filter(pl.col("is_ghost") == True)["candidate_id"].to_list()
                )
        exact_ids = build_exact_recall_ranked_list(candidates, ghost_ids=ghost_ids)
        rrf_scores = _rrf([
            dense_ids_v1, dense_ids_recsys, dense_ids_eval, sparse_ids, sparse_ids_bm25, exact_ids
        ])
        print(f"Added exact recall lane: {len(exact_ids)} candidates")
    
    retrieved = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:constants.RRF_PRECOMPUTE_TOPK]
    
    df = pd.DataFrame(retrieved, columns=["candidate_id", "rrf_score"])
    df.to_parquet(constants.RETRIEVAL_SCORES_PARQUET)
    print(f"RRF retrieval saved to {constants.RETRIEVAL_SCORES_PARQUET} ({len(df)} candidates)")


def run_phase_1d_recall_rescue(candidates):
    """
    Widen an existing retrieval_scores.parquet with the exact recall lane.

    Intended for --skip-embed experiments: keep existing BGE/FAISS/BM25 retrieval
    scores, add a CPU-cheap all-corpus lexical rescue list, then re-extract
    features for the widened pool.
    """
    print("\n--- Phase 1d: High-Recall Rescue (existing RRF + exact lane) ---")
    base_path = constants.BASE_RETRIEVAL_SCORES_PARQUET
    if not os.path.exists(base_path):
        if not os.path.exists(constants.RETRIEVAL_SCORES_PARQUET):
            print(f"Warning: {constants.RETRIEVAL_SCORES_PARQUET} not found. Skipping recall rescue.")
            return
        shutil.copyfile(constants.RETRIEVAL_SCORES_PARQUET, base_path)
        print(f"Created base retrieval snapshot: {base_path}")

    if not os.path.exists(base_path):
        print(f"Warning: {base_path} not found. Skipping recall rescue.")
        return

    ghost_ids = set()
    if os.path.exists(constants.CANDIDATE_FLAGS_PARQUET):
        flags_df = pl.read_parquet(constants.CANDIDATE_FLAGS_PARQUET)
        if "is_ghost" in flags_df.columns:
            ghost_ids = set(
                flags_df.filter(pl.col("is_ghost") == True)["candidate_id"].to_list()
            )

    retrieval_df = pl.read_parquet(base_path)
    base_ids = retrieval_df.sort("rrf_score", descending=True)["candidate_id"].to_list()
    exact_ids = build_exact_recall_ranked_list(candidates, ghost_ids=ghost_ids)

    fused_scores = _rrf([base_ids, exact_ids])
    retrieved = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:constants.RRF_PRECOMPUTE_TOPK]
    df = pd.DataFrame(retrieved, columns=["candidate_id", "rrf_score"])
    df.to_parquet(constants.RETRIEVAL_SCORES_PARQUET)
    print(
        f"High-recall retrieval saved to {constants.RETRIEVAL_SCORES_PARQUET} "
        f"({len(df)} candidates; exact lane={len(exact_ids)})"
    )


# ---------------------------------------------------------------------------
# Phase 1c: Feature Extraction
# ---------------------------------------------------------------------------

def run_phase_1c_features(candidates, is_skip_embed=False, force_all=False):
    print("\n--- Phase 1c: Feature Extraction ---")
    
    # Read flags
    flags_df = pl.read_parquet(constants.CANDIDATE_FLAGS_PARQUET)
    flags_dict = {row["candidate_id"]: row for row in flags_df.to_dicts()}
    
    # Determine which candidates to extract
    if force_all:
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

def run_phase_1e_cross_encoder(candidates, model, retrieval_path=None, output_path=None):
    print("\n--- Phase 1e: Cross-Encoder ---")
    _retrieval_path = retrieval_path or constants.RETRIEVAL_SCORES_PARQUET
    retrieval_df = pl.read_parquet(_retrieval_path)
    top_ids = retrieval_df["candidate_id"].to_list()  # use all rows in the parquet (partition-aware)
    top_set = set(top_ids)
    
    # CE query loaded directly from docs/ce_query_profile.md.
    # No YAML generation, no keyword lists, no JD anchors, no HyDE text.
    with open(constants.CE_QUERY_PROFILE_MD, "r", encoding="utf-8") as _f:
        jd_v1 = _f.read().strip()
    
    pairs = []
    pair_cids = []
    for c in candidates:
        if c.get("candidate_id") in top_set:
            text = build_profile_text(c)
            pairs.append([jd_v1, text])
            pair_cids.append(c.get("candidate_id"))
            
    print(f"Scoring {len(pairs)} candidates with cross-encoder...")
    # Process in chunks to avoid OOM
    batch_size = constants.CE_BATCH_SIZE
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size), mininterval=15.0, maxinterval=60.0, smoothing=0.0):
        batch = pairs[i:i + batch_size]
        batch_scores = model.compute_score(batch, max_length=constants.CE_MAX_LENGTH)
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
        
    df = pd.DataFrame({"candidate_id": pair_cids, "ce_score": scores})
    _output_path = output_path or constants.CROSS_ENCODER_SCORES_PARQUET
    os.makedirs(os.path.dirname(_output_path) or ".", exist_ok=True)
    df.to_parquet(_output_path)
    print(f"Cross-encoder scores saved to {_output_path} ({len(df)} candidates)")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline Corpus Preprocessing Pipeline")
    parser.add_argument("--candidates", type=str, default=constants.CANDIDATES_JSONL, help="Path to candidates.jsonl")
    parser.add_argument("--skip-embed", action="store_true", help="Skip heavy embedding phases for local testing (runs honeypot & features only)")
    parser.add_argument("--only-cross-encoder", action="store_true", help="Only refresh cross-encoder scores for the current retrieval pool")
    parser.add_argument("--retrieval", type=str, default=None,
                        help="Path to retrieval parquet (partition support). Default: constants.RETRIEVAL_SCORES_PARQUET")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for CE output parquet (partition support). Default: constants.CROSS_ENCODER_SCORES_PARQUET")
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

    if args.only_cross_encoder:
        print("\nRunning only Phase 1e Cross-Encoder refresh")
        if args.retrieval:
            print(f"  Retrieval source: {args.retrieval}")
        if args.output:
            print(f"  Output target:    {args.output}")
        ce_model = get_cross_encoder()
        run_phase_1e_cross_encoder(
            candidates, ce_model,
            retrieval_path=args.retrieval,
            output_path=args.output,
        )
        print("\nCross-Encoder Refresh Complete!")
        return
    
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
        run_phase_1d_rrf(candidates)
    elif not args.sample:
        # Reuse existing dense/sparse/BM25 retrieval artifacts, then add the
        # CPU-cheap exact recall lane. This lets us test upper-funnel recall
        # without recomputing BGE embeddings.
        run_phase_1d_recall_rescue(candidates)
    else:
        print("\nSkipping Phase 1d recall rescue for --sample --skip-embed")
        
    # Phase 1c (Features) - Works with or without embeddings
    run_phase_1c_features(candidates, is_skip_embed=args.skip_embed, force_all=args.sample)
    
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
