## 9. Phase 5 — Behavioral Re-ranking + Penalization

### 9.1 Reachability multiplier

Behavioral signals are applied **only here in Phase 5** — not in Phase 1 retrieval. Retrieval stays 100% relevance-based. Senior engineers are disproportionately `open_to_work=False` (they're employed and headhunted). Removing them at retrieval time risks dropping strong passive candidates that the hidden ground truth scores highly.

**Failure mode of early filtering:** unrecoverable. **Failure mode of late down-weighting:** a slightly over-ranked passive candidate, survivable.

```python
def reachability_multiplier(signals: dict) -> float:
    """
    Compound multiplier from behavioral signals.
    Worst-case dead profile: ~0.57x — still beats a mediocre active candidate.
    Strong passive candidate with great skills will still surface.
    """
    from datetime import date
    mult = 1.0

    last_active_str = signals.get("last_active_date")
    if last_active_str:
        days_inactive = (date.today() - date.fromisoformat(last_active_str)).days
        if days_inactive > 540:
            mult *= 0.60
        elif days_inactive > 270:
            mult *= 0.75

    if not signals.get("open_to_work_flag", True):
        mult *= 0.85

    if signals.get("recruiter_response_rate", 1.0) < 0.10:
        mult *= 0.90

    return mult
```

### 9.2 Notice period modifier

```python
def notice_modifier(days) -> float:
    if days is None:
        return 1.00  # Fail open / default
    if days <= 30:
        return 1.00  # JD: ideal, can buy out up to 30 days
    elif days <= 60:
        return 0.95
    elif days <= 90:
        return 0.90  # JD: "bar gets higher" at 30+
    else:
        return 0.75  # JD: "significant concern" (>90 days) per contract
```

### 9.3 Location modifier

```python
PUNE_NOIDA_CITIES = {"pune", "noida", "greater noida", "delhi", "new delhi",
                      "gurugram", "gurgaon", "faridabad", "ghaziabad"}
JD_WELCOME_CITIES = {"hyderabad", "mumbai"}
INDIA_ADJACENT = {"bangalore", "bengaluru", "chennai", "kolkata",
                   "ahmedabad", "indore", "jaipur", "chandigarh", "kochi"}

def location_modifier(behavioral) -> float:
    location = behavioral["location"]
    country = behavioral["country"]
    willing = behavioral["willing_to_relocate"]

    if any(city in location for city in PUNE_NOIDA_CITIES):
        return 1.0
    if any(city in location for city in JD_WELCOME_CITIES):
        return 1.00 if willing else 0.98
    if any(city in location for city in INDIA_ADJACENT):
        return 0.98 if willing else 0.95
    if country == "india" and willing:
        return 0.95
    if country == "india":
        return 0.92
    if willing:
        return 0.90
    return 0.85
```

### 9.4 Social proof boost

```python
def social_proof_boost(behavioral) -> float:
    """
    Additive boost from Redrob platform signals not already captured by multipliers.
    Uses 9 of the 23 redrob_signals fields. Cap at 0.12 so no single cluster dominates.
    """
    boost = 0.0

    # --- Market validation (other recruiters already found this person valuable) ---
    if behavioral["github_activity_score"] > 60:
        boost += 3.0  # JD: external validation valued; open-source contributions
    if behavioral["saved_by_recruiters_30d"] > 5:
        boost += 4.0  # Human-curated: other recruiters are already shortlisting them
    if behavioral.get("profile_views_received_30d", 0) > 20:
        boost += 1.0  # Passive market interest — searched for and clicked on

    # --- Engagement quality (serious about the job search) ---
    if behavioral["endorsements_received"] > 20:
        boost += 1.0  # Peer credibility signal
    if behavioral.get("interview_completion_rate", 0) > 0.80:
        boost += 2.0  # Shows up and follows through — predictive of offer conversion
    if behavioral.get("offer_acceptance_rate", -1) > 0.70:
        boost += 1.0  # When they receive offers they accept them — not just browsing

    # --- Profile credibility ---
    if behavioral.get("profile_completeness_score", 0) > 80:
        boost += 1.0  # Actively managing profile = genuinely in the market
    if behavioral["linkedin_connected"]:
        boost += 1.0  # Basic platform legitimacy

    # --- Response speed (availability complement) ---
    avg_rt = behavioral.get("avg_response_time_hours", 24.0)
    if avg_rt <= 4.0 and behavioral["recruiter_response_rate"] >= 0.60:
        boost += 1.0  # Fast AND responsive — highest-reachability signal

    return min(boost, 12.0)  # Cap: no single signal cluster should dominate final score
```

### 9.5 Seniority modifier

```python
def seniority_modifier(bucket_c) -> float:
    """
    Applies the seniority soft window from Bucket C.
    JD: "5-9 years is a range, not a requirement. We'll seriously consider candidates
    outside the band if other signals are strong."
    Returns the seniority_score computed in Phase 3 — no additional logic needed.
    """
    return bucket_c.get("seniority_score", 1.0)
```

### 9.6 Soft penalties

All penalties are **soft multipliers**. The JD uses language like "probably not move forward" and "the bar gets higher" — not "automatic reject". Only honeypots and ghosts are heavily penalized (multiplied by 0.01). Everything else is a downward nudge that strong technical evidence can overcome.

> **Validate harsh penalties before trusting them.** The multipliers below (`consulting_only ×0.4`, `research_only ×0.40`, `langchain_only_flag ×0.45`) are calibrated from JD language but have not yet been tested against the actual labeled set. If Redrob intentionally inserted edge-case candidates — e.g. a strong researcher with genuine product exposure, or a LangChain practitioner with a deep pre-LLM ML background — these will over-penalise real fits. Run Phase 7 validation against `metadata/validation_set.json` first. In particular: confirm that no Tier 3+ labeled candidate is being hard-penalised by a flag that misread their profile.

```python
def soft_penalties(bucket_c, flags, behavioral) -> float:
    multiplier = 1.0

    # Consistency score: skill-career mismatch multiplier computed from Phase 1B flags.
    # Score 1.0 = no penalty. Score 0.30 = heavy mismatch (keyword stuffer).
    # Drops 0.15 per contradiction.
    contradictions = flags.get("contradiction_skill_duration", 0) + flags.get("contradiction_assessment", 0)
    consistency_score = max(0.30, 1.0 - (0.15 * contradictions))
    multiplier *= consistency_score

    # Title velocity: switched every ~1.5 years across 3+ jobs
    # JD: "not a fit" — strong signal but softened to 0.80;
    # many startup engineers legitimately switch frequently.
    if bucket_c["title_velocity_flag"]:
        multiplier *= 0.80

    # Code stopped: architect/VP/Director with yoe > 8
    # JD: "probably not move forward" — soft, not hard.
    if bucket_c["code_stopped"]:
        multiplier *= 0.75

    # LangChain-only AI experience under 12 months, no pre-LLM ML background
    # JD: "probably not move forward, unless substantial pre-LLM ML production experience"
    if bucket_c.get("langchain_only_flag"):
        multiplier *= 0.45

    # Remote-only preference for a hybrid role
    pref_mode = behavioral.get("preferred_work_mode", "").lower().strip()
    if pref_mode in ("remote", "wfh", "work from home"):
        multiplier *= 0.85

    # Research-only background — JD: "will not move forward" — strongest language
    if flags.get("research_only"):
        multiplier *= 0.40

    # Wrong domain (CV/speech without NLP/IR)
    if flags.get("wrong_domain"):
        multiplier *= 0.50


    # Closed-source only for 5+ years without external validation (GitHub, papers, talks)
    if bucket_c.get("closed_source_flag"):
        multiplier *= 0.80

    return multiplier
```

### 9.7 Final score assembly

```python
def compute_final_score(candidate_data) -> float:
    phase4_score = candidate_data["final_phase4_score"]
    behavioral = candidate_data["behavioral"]
    bucket_a = candidate_data.get("bucket_a", {})
    bucket_b = candidate_data["bucket_b"]
    bucket_c = candidate_data["bucket_c"]
    flags = candidate_data["flags"]

    # Honeypot scoring — 0.01 hedge
    # Proof of bounds: Since phase4_score is strictly capped at 100.0, a 0.01 multiplier 
    # guarantees a maximum score of 1.0. This ensures the candidate drops completely 
    # out of the Top 100, since average valid candidates will score 40.0+.
    # Absolute zero is avoided so false positives remain discoverable.
    # Note: Early return ensures the 0.25 safety floor below does NOT protect honeypots.
    if flags.get("impossible_flag") or flags.get("suspicious_flag"):
        return phase4_score * 0.01

    reachability_mult = reachability_multiplier(behavioral)
    penalty_mult = soft_penalties(bucket_c, flags, behavioral)

    # Logistical signals: notice period, location, seniority, writing culture.
    # Grouped and floor-capped so no single operational signal collapses the score.
    notice_mult = notice_modifier(behavioral["notice_period_days"])
    loc_mult = location_modifier(behavioral)
    seniority_mult = seniority_modifier(bucket_c)
    writing_mult = bucket_b.get("writing_signal", 1.0)

    logistical_mult = notice_mult * loc_mult * seniority_mult * writing_mult
    logistical_mult = max(logistical_mult, 0.75)  # Floor: logistics cannot reduce score by >25%

    # Combined multiplier floor: prevents the full chain (availability × penalties × logistics)
    # from collapsing a strong technical score into near-zero. Strong technical fit must
    # always be able to show through — no variable cluster should dominate alone.
    combined_mult = reachability_mult * penalty_mult * logistical_mult
    combined_mult = max(combined_mult, 0.25)  # Floor: maximum total reduction is 75%

    # Soft honeypot score compound penalty (for non-flagged suspicious profiles)
    honeypot_score = flags.get("honeypot_score", 0.0)
    honeypot_mult = 1.0 - (honeypot_score * 0.40)  # Range: 1.0 down to 0.60
    combined_mult *= honeypot_mult

    # Additive bonuses — reward without gating.
    # 90-day alignment: JD describes 3 milestones (audit retrieval, ship v2 ranker, build eval
    # framework). Moved from multiplicative to additive: being able to execute all 3 on day 1
    # is a bonus signal, not a disqualifier. Missing milestone evidence ≠ wrong hire.
    product_ratio = bucket_b.get("product_ratio", 0.5)
    ninety_day_alignment = compute_ninety_day_alignment(bucket_a, product_ratio)
    ninety_day_bonus = 8.0 * ninety_day_alignment  # Range: 0.0 to +8.0

    # Platform signals: market validation, engagement quality, profile credibility.
    # Uses 9 of the 23 redrob_signals fields not already captured in multipliers above.
    social_boost = social_proof_boost(behavioral)  # Range: 0.0 to +12.0 (capped)

    # Final formula:
    # Multiplicative chain: technical fit × strong behavioral gates × logistical group (capped)
    # Additive bonuses: 90-day milestone readiness + Redrob platform signal cluster
    # No single variable can move the score by more than ~75% on its own — mix of signals.
    final = (
        phase4_score
        * combined_mult
        + ninety_day_bonus
        + social_boost
    )

    # Floor protection: if penalties drop the score near zero, ensure it hits 0.0 to drop out of ranking
    if penalty_mult < 0.20:
        return 0.0

    # Score range bounding (max 120.0)
    final = min(final, 120.0)

    return round(float(final), 6)


# --- Rank assignment with deterministic tie-breaking ---
# Spec §3: "If two candidates have the same score, you must still assign unique ranks.
# Break score ties deterministically using a secondary signal from your model,
# or by candidate_id ascending."
#
# Sort key: (-final_score, candidate_id)
# candidate_id is a string (CAND_XXXXXXX); ascending lexicographic order is deterministic.
def assign_ranks(scored_candidates: list[dict]) -> list[dict]:
    sorted_cands = sorted(
        scored_candidates,
        key=lambda c: (-c["final_score"], c["candidate_id"])
    )
    for rank, c in enumerate(sorted_cands, start=1):
        c["rank"] = rank
        
    # Validation constraint check (non-increasing score)
    for i in range(1, len(sorted_cands)):
        assert sorted_cands[i]["final_score"] <= sorted_cands[i-1]["final_score"], "Score sorting failed"
        
    return sorted_cands
```

---

