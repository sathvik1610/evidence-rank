"""
test_features.py — Phase Review test for src/features.py
Run: python test_features.py
"""
import sys
import json
sys.path.insert(0, ".")
from src.features import (
    extract_features, compute_product_ratio, score_skill_bucket,
    score_career_quality, score_fit_gaps, _build_career_text, extract_behavioral
)

# ── Load sample data ──────────────────────────────────────────────────────
with open("Resources/sample_candidates.json") as f:
    CANDIDATES = json.load(f)

print(f"Loaded {len(CANDIDATES)} sample candidates\n")

PASS = 0
FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS: {name}")
        PASS += 1
    else:
        print(f"  FAIL: {name}" + (f" — {msg}" if msg else ""))
        FAIL += 1

# ── Test 1: No crashes on all 50 sample candidates ────────────────────────
print("=== Test 1: Smoke test — extract_features on all 50 candidates ===")
crashes = []
for i, cand in enumerate(CANDIDATES):
    try:
        flags = {
            "product_ratio": compute_product_ratio(cand),
            "consulting_only": False,
            "research_only": False,
            "wrong_domain": False,
        }
        f = extract_features(cand, flags)
        assert "candidate_id" in f
        assert "retrieval_search" in f
        assert "snippets_json" in f
        json.loads(f["snippets_json"])  # Must be valid JSON
    except Exception as e:
        crashes.append(f"Cand {i}: {e}")
check("No crashes on 50 candidates", len(crashes) == 0, str(crashes[:3]))

# ── Test 2: Score ranges ──────────────────────────────────────────────────
print("\n=== Test 2: Score range validation ===")
for i, cand in enumerate(CANDIDATES[:20]):
    flags = {"product_ratio": compute_product_ratio(cand), "consulting_only": False, "research_only": False, "wrong_domain": False}
    f = extract_features(cand, flags)
    pbs = f["product_builder_score"]
    rs  = f["retrieval_search"]
    ss  = f["seniority_score"]
    nd  = f["ninety_day_alignment"]
    check(f"  cand[{i}] product_builder in [0,1]", 0.0 <= pbs <= 1.0, str(pbs))
    check(f"  cand[{i}] retrieval_search in [0,3.5]", 0.0 <= rs <= 3.5, str(rs))
    check(f"  cand[{i}] seniority_score in valid set", ss in {0.75, 0.90, 0.95, 1.00}, str(ss))
    check(f"  cand[{i}] ninety_day_alignment in [0,1]", 0.0 <= nd <= 1.0, str(nd))

# ── Test 3: Realistic candidate — strong IR background ────────────────────
print("\n=== Test 3: Realistic candidate — strong IR/search background ===")
strong_cand = {
    "candidate_id": "TEST_STRONG",
    "profile": {"current_title": "ML Engineer", "years_of_experience": 6.0, "location": "Bangalore", "country": "India"},
    "career_history": [
        {
            "title": "Senior ML Engineer",
            "company": "Swiggy",
            "industry": "Food Tech",
            "duration_months": 36,
            "is_current": True,
            "description": "Built and shipped production retrieval system using FAISS and BM25 hybrid search. Deployed dense retrieval serving 10M+ queries per day. Implemented NDCG evaluation framework for offline A/B testing. Real users depend on this system daily."
        },
        {
            "title": "ML Engineer",
            "company": "Razorpay",
            "industry": "Fintech",
            "duration_months": 30,
            "is_current": False,
            "description": "Designed recommendation system and ranking pipeline. Used XGBoost learning-to-rank. Shipped to production users at scale. Built vector search infrastructure."
        }
    ],
    "skills": [
        {"name": "FAISS", "proficiency": "expert", "duration_months": 24, "endorsements": 5},
        {"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 12},
        {"name": "NDCG", "proficiency": "advanced", "duration_months": 18, "endorsements": 3},
    ],
    "redrob_signals": {
        "last_active_date": "2026-05-15", "open_to_work_flag": True,
        "recruiter_response_rate": 0.75, "github_activity_score": 72,
        "notice_period_days": 30, "applications_submitted_30d": 2
    },
    "education": []
}
flags_strong = {"product_ratio": 1.0, "consulting_only": False, "research_only": False, "wrong_domain": False}
f_strong = extract_features(strong_cand, flags_strong)

check("Strong cand: retrieval_search >= 2", f_strong["retrieval_search"] >= 2.0, str(f_strong["retrieval_search"]))
check("Strong cand: eval_framework >= 1", f_strong["eval_framework"] >= 1.0, str(f_strong["eval_framework"]))
check("Strong cand: python_coding >= 1", f_strong["python_coding"] >= 1.0, str(f_strong["python_coding"]))
check("Strong cand: product_builder > 0.5", f_strong["product_builder_score"] > 0.5, str(f_strong["product_builder_score"]))
check("Strong cand: sys_experience_score == 1.0", f_strong["sys_experience_score"] == 1.0, str(f_strong["sys_experience_score"]))
check("Strong cand: seniority_score == 1.0 (6 YoE)", f_strong["seniority_score"] == 1.0, str(f_strong["seniority_score"]))
check("Strong cand: depth_signal == 1.0 (2 retrieval roles)", f_strong["depth_signal"] == 1.0, str(f_strong["depth_signal"]))

# ── Test 4: Pure consulting candidate ────────────────────────────────────
print("\n=== Test 4: Consulting-only candidate ===")
consulting_cand = {
    "candidate_id": "TEST_CONSULT",
    "profile": {"current_title": "Senior Consultant", "years_of_experience": 8.0, "location": "Mumbai", "country": "India"},
    "career_history": [
        {"title": "Senior Consultant", "company": "TCS", "industry": "IT Services",
         "duration_months": 48, "is_current": True, "description": "Worked on client AI projects using Python and machine learning."},
        {"title": "Analyst", "company": "Infosys", "industry": "IT Services",
         "duration_months": 48, "is_current": False, "description": "Built dashboards for enterprise clients."},
    ],
    "skills": [{"name": "Python", "proficiency": "intermediate", "duration_months": 60}],
    "redrob_signals": {"last_active_date": "2026-06-01", "open_to_work_flag": True},
    "education": []
}
product_ratio_consult = compute_product_ratio(consulting_cand)
flags_consult = {"product_ratio": product_ratio_consult, "consulting_only": True, "research_only": False, "wrong_domain": False}
f_consult = extract_features(consulting_cand, flags_consult)

check("Consulting: product_ratio == 0.0", product_ratio_consult == 0.0, str(product_ratio_consult))
check("Consulting: consulting_flag == True", f_consult["consulting_flag"] == True)
check("Consulting: product_builder_score penalized (<0.2)", f_consult["product_builder_score"] < 0.2, str(f_consult["product_builder_score"]))

# ── Test 5: LangChain-only flag ───────────────────────────────────────────
print("\n=== Test 5: LangChain-only candidate ===")
lc_cand = {
    "candidate_id": "TEST_LC",
    "profile": {"current_title": "AI Engineer", "years_of_experience": 1.5, "location": "Pune", "country": "India"},
    "career_history": [
        {"title": "AI Engineer", "company": "Startup", "industry": "Tech",
         "duration_months": 18, "is_current": True,
         "description": "Used LangChain and OpenAI API to build chatbot demos. Wrapped GPT-4 for document Q&A. Built LangChain pipelines."}
    ],
    "skills": [
        {"name": "LangChain", "proficiency": "advanced", "duration_months": 6},
        {"name": "OpenAI", "proficiency": "intermediate", "duration_months": 5},
    ],  # Total AI skill months = 11 (< 12 threshold → langchain_only_flag should fire)
    "redrob_signals": {},
    "education": []
}
flags_lc = {"product_ratio": 0.8, "consulting_only": False, "research_only": False, "wrong_domain": False}
f_lc = extract_features(lc_cand, flags_lc)
check("LangChain-only: langchain_only_flag == True", f_lc["langchain_only_flag"] == True, str(f_lc["langchain_only_flag"]))
check("LangChain-only: seniority_score < 1.0 (1.5 YoE)", f_lc["seniority_score"] < 1.0, str(f_lc["seniority_score"]))

# ── Test 6: Sentinel safety ───────────────────────────────────────────────
print("\n=== Test 6: Adversarial — empty candidate (all fields missing) ===")
empty_cand = {"candidate_id": "EMPTY", "profile": {}, "career_history": [], "skills": [], "redrob_signals": {}, "education": []}
flags_empty = {"product_ratio": 0.5, "consulting_only": False, "research_only": False, "wrong_domain": False}
try:
    f_empty = extract_features(empty_cand, flags_empty)
    check("Empty candidate: no crash", True)
    check("Empty candidate: product_builder_score >= 0", f_empty["product_builder_score"] >= 0.0)
    check("Empty candidate: seniority_score == 1.0 (YoE missing sentinel)", f_empty["seniority_score"] == 1.0, str(f_empty["seniority_score"]))
    check("Empty candidate: langchain_only_flag == False", f_empty["langchain_only_flag"] == False)
    check("Empty candidate: beh_offer_acceptance_rate == -1", f_empty["beh_offer_acceptance_rate"] == -1)
except Exception as e:
    check("Empty candidate: no crash", False, str(e))

# ── Test 7: Adversarial — candidate with future YoE (e.g., 30 years) ─────
print("\n=== Test 7: Adversarial — extreme YoE (30 years) ===")
old_cand = {
    "candidate_id": "OLD_GUY",
    "profile": {"current_title": "CTO", "years_of_experience": 30.0, "location": "Delhi", "country": "India"},
    "career_history": [{"title": "CTO", "company": "MegaCorp", "industry": "Tech", "duration_months": 120, "is_current": True, "description": "Led AI strategy."}],
    "skills": [], "redrob_signals": {}, "education": []
}
flags_old = {"product_ratio": 0.5, "consulting_only": False, "research_only": False, "wrong_domain": False}
f_old = extract_features(old_cand, flags_old)
check("Extreme YoE: seniority_score == 0.90 (30 >= 13)", f_old["seniority_score"] == 0.90, str(f_old["seniority_score"]))
check("Extreme YoE: code_stopped == True (CTO + YoE>8)", f_old["code_stopped"] == True, str(f_old["code_stopped"]))

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Results: {PASS} PASSED, {FAIL} FAILED")
if FAIL > 0:
    print("FIX ALL FAILURES before proceeding to preprocess.py")
    sys.exit(1)
else:
    print("Phase Review PASSED. features.py is verified.")
