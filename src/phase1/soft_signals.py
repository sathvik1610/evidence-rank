from typing import Dict, Any
import datetime

def extract_soft_signals(candidate: Dict[str, Any], chrono_months: float) -> Dict[str, Any]:
    """
    Evaluates Phase 1B (Consistency) and Phase 1C (Disqualifiers).
    Returns a flat dictionary of signals for candidate_flags.parquet.
    """
    signals = {}
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    
    # Phase 1B: Consistency Penalties
    c_skill = 0
    c_assess = 0
    
    for skill in skills:
        duration = skill.get("duration_months", 0)
        if duration > chrono_months + 48:
            c_skill += 1
            
    for skill in skills:
        if skill.get("proficiency") in ("expert", "advanced"):
            score = assessment_scores.get(skill["name"])
            if score is not None and score < 40:
                c_assess += 1
                
    signals["contradiction_skill_duration"] = c_skill
    signals["contradiction_assessment"] = c_assess

    # Phase 1C: Disqualifiers
    titles_lower = [r.get("title", "").lower() for r in career]
    desc_text = " ".join(r.get("description", "").lower() for r in career)
    skills_lower = [s.get("name", "").lower() for s in skills]

    CONSULTING_FIRMS = {
        "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
        "ltimindtree", "l&t infotech", "niit technologies", "zensar", "mastech",
        "syntel", "kpit", "cyient", "birlasoft", "persistent systems"
    }

    total_months = sum(r.get("duration_months", 0) for r in career)
    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if any(firm in r.get("company", "").lower() for firm in CONSULTING_FIRMS)
        or r.get("industry", "").lower() in ("it services", "consulting", "outsourcing")
    )
    product_ratio = 1.0 - (consulting_months / total_months) if total_months > 0 else 0.0
    signals["product_ratio"] = round(product_ratio, 4)
    signals["consulting_only"] = (product_ratio == 0.0)

    engineering_titles = {"engineer", "developer", "data scientist", "applied scientist", "architect", "lead", "head"}
    research_titles = {"researcher", "research scientist", "phd", "postdoc", "intern"}
    has_engineering = any(t in " ".join(titles_lower) for t in engineering_titles)
    has_only_research = not has_engineering and any(t in " ".join(titles_lower) for t in research_titles)
    signals["research_only"] = has_only_research
    
    cv_speech_terms = {"computer vision", "opencv", "yolo", "object detection", "speech recognition", "tts", "asr", "robotics"}
    nlp_ir_terms = {"nlp", "retrieval", "ranking", "recommendation", "search", "embedding", "information retrieval"}
    has_cv_speech = any(t in desc_text or any(t in s for s in skills_lower) for t in cv_speech_terms)
    has_nlp_ir = any(t in desc_text or any(t in s for s in skills_lower) for t in nlp_ir_terms)
    signals["wrong_domain"] = has_cv_speech and not has_nlp_ir

    return signals

def check_ghost(candidate: Dict[str, Any], reference_date: datetime.date = datetime.date(2026, 6, 1)) -> bool:
    """Pre-filter inactive candidates."""
    signals = candidate.get("redrob_signals", {})
    last_active_str = signals.get("last_active_date")
    if not last_active_str:
        return False
    
    try:
        last_active = datetime.date.fromisoformat(last_active_str)
        days_inactive = (reference_date - last_active).days
    except ValueError:
        return False
        
    return (
        days_inactive > 365
        and signals.get("recruiter_response_rate", 1.0) < 0.05
        and not signals.get("open_to_work_flag", True)
        and signals.get("applications_submitted_30d", 1) == 0
    )
