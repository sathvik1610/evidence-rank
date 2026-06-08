import sys
sys.path.insert(0, ".")

import polars as pl

from src.reranker import normalize_ce_scores


def test_normalize_ce_scores_converts_raw_logits_to_0_100():
    df = pl.DataFrame({
        "candidate_id": ["A", "B", "C"],
        "ce_score": [-8.0, -2.0, 4.0],
    })

    out = normalize_ce_scores(df)
    scores = out["ce_score"].to_list()

    assert scores == [0.0, 50.0, 100.0]


def test_normalize_ce_scores_keeps_existing_0_100_scores():
    df = pl.DataFrame({
        "candidate_id": ["A", "B", "C"],
        "ce_score": [12.0, 55.0, 91.0],
    })

    out = normalize_ce_scores(df)

    assert out["ce_score"].to_list() == [12.0, 55.0, 91.0]


def test_normalize_ce_scores_handles_flat_raw_scores():
    df = pl.DataFrame({
        "candidate_id": ["A", "B"],
        "ce_score": [-3.0, -3.0],
    })

    out = normalize_ce_scores(df)

    assert out["ce_score"].to_list() == [50.0, 50.0]
