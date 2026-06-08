import sys
sys.path.insert(0, ".")
import pytest
from src.jd_intelligence import (
    _norm_term, _term_to_regex, _dedupe, load_jd_contract,
    get_rule_patterns, get_condition_values, get_multiplier_value,
    get_location_bands, build_feature_contract, build_jd_intelligence
)
import constants

def test_helper_functions():
    # Normalization
    assert _norm_term("  Vector   Search  ") == "vector search"
    assert _norm_term("fine-tuning\n") == "fine-tuning"
    
    # Term to Regex
    regex = _term_to_regex("vector search")
    assert "vector" in regex
    assert "search" in regex
    assert r"[\s._/-]+" in regex

    # Deduplication
    assert _dedupe(["a", "A", "b", " a"]) == ["a", "b"]

def test_load_jd_contract():
    contract = load_jd_contract(constants.JD_CONTRACT_YAML)
    assert isinstance(contract, dict)
    assert "metadata" in contract
    assert "extraction_rules" in contract

def test_get_rule_patterns():
    contract = load_jd_contract(constants.JD_CONTRACT_YAML)
    patterns = get_rule_patterns(contract, "core_search_and_retrieval", as_regex=False)
    assert "bm25" in patterns
    assert "faiss" in patterns

def test_get_condition_values():
    contract = load_jd_contract(constants.JD_CONTRACT_YAML)
    values = get_condition_values(contract, "pure_research_penalty", "profile.current_title")
    assert "research scientist" in values

def test_get_multiplier_value():
    contract = load_jd_contract(constants.JD_CONTRACT_YAML)
    val = get_multiplier_value(contract, "pure_research_penalty", default=1.0)
    assert val == 0.20

def test_get_location_bands():
    contract = load_jd_contract(constants.JD_CONTRACT_YAML)
    bands = get_location_bands(contract)
    assert "noida" in bands["preferred"]
    assert "hyderabad" in bands["welcome"]

def test_build_feature_contract():
    fc = build_feature_contract(constants.JD_CONTRACT_YAML)
    assert "target_skills" in fc
    assert "retrieval_patterns" in fc
    assert "seniority_bands" in fc
    assert "location_bands" in fc

def test_build_jd_intelligence():
    intel = build_jd_intelligence(constants.JD_CONTRACT_YAML, constants.JD_TEXT)
    assert "config" in intel
    assert "keywords" in intel
    assert "queries" in intel
    assert "cross_encoder_query" in intel
