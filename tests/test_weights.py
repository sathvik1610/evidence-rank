import sys
sys.path.insert(0, ".")
import pytest
from src.weights import _flatten, W

def test_flatten():
    nested = {
        "scoring": {
            "weight": 0.5,
            "sub": {
                "sub_weight": 0.1
            }
        },
        "simple": 1
    }
    expected = {
        "scoring.weight": 0.5,
        "scoring.sub.sub_weight": 0.1,
        "simple": 1
    }
    assert _flatten(nested) == expected

def test_singleton_weights():
    assert isinstance(W, dict)
    assert len(W) > 0
    # Verify standard expected keys
    assert "scoring.must_have_weight" in W
    assert "behavioral.inactive_heavy_days" in W
    assert "social_proof.github_threshold" in W
    # Verify values are numeric
    assert isinstance(W["scoring.must_have_weight"], float)
    assert isinstance(W["behavioral.inactive_heavy_days"], int)
