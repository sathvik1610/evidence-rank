import polars as pl

from src.reranker import normalize_ce_scores


def test_normalize_raw_ce_logits():
    df = pl.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "ce_score": [-8.0, -2.0, 4.0],
    })

    out = normalize_ce_scores(df)
    scores = out["ce_score"].to_list()

    assert scores == [0.0, 50.0, 100.0]


def test_normalize_ce_passthrough_when_already_scaled():
    df = pl.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "ce_score": [12.0, 55.0, 91.0],
    })

    out = normalize_ce_scores(df)

    assert out["ce_score"].to_list() == [12.0, 55.0, 91.0]


def test_normalize_ce_constant_scores_to_neutral():
    df = pl.DataFrame({
        "candidate_id": ["a", "b"],
        "ce_score": [-3.0, -3.0],
    })

    out = normalize_ce_scores(df)

    assert out["ce_score"].to_list() == [50.0, 50.0]
