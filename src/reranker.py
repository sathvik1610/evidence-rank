"""
src/reranker.py — Phase 4: Cross-Encoder Merge (Runtime)

Loads the offline-computed bge-reranker-v2-m3 scores and merges them with the
handcrafted core_score to produce final_phase4_score.
Weights sourced from weights.yaml via W.
"""

import polars as pl
import os
import constants
from src.weights import W


def normalize_ce_scores(ce_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize legacy raw cross-encoder logits to the 0-100 core-score scale.

    Newer preprocess runs may already save normalized CE scores. Older artifacts
    often store raw FlagReranker logits around -8..3 under the same ce_score
    column. Runtime ranking must handle both formats because artifacts can be
    regenerated independently from code changes.
    """
    if "ce_score" not in ce_df.columns or ce_df.is_empty():
        return ce_df

    stats = ce_df.select(
        pl.col("ce_score").min().alias("min_score"),
        pl.col("ce_score").max().alias("max_score"),
    ).row(0, named=True)
    min_score = stats["min_score"]
    max_score = stats["max_score"]

    if min_score is None or max_score is None:
        return ce_df

    # Already on a 0-100-ish scale.
    if min_score >= 0.0 and max_score <= 100.0 and max_score > 10.0:
        return ce_df

    if max_score == min_score:
        return ce_df.with_columns(pl.lit(50.0).alias("ce_score"))

    return ce_df.with_columns(
        (((pl.col("ce_score") - min_score) / (max_score - min_score)) * 100.0)
        .clip(0.0, 100.0)
        .alias("ce_score")
    )


def merge_cross_encoder_scores(scored_df: pl.DataFrame) -> pl.DataFrame:
    """
    Left-joins the precomputed cross-encoder scores and applies the
    handcrafted/CE weighted merge from weights.yaml.
    Gracefully falls back to core_score if CE score is missing.
    """
    if not os.path.exists(constants.CROSS_ENCODER_SCORES_PARQUET):
        print(f"Warning: {constants.CROSS_ENCODER_SCORES_PARQUET} not found. Skipping CE merge.")
        return scored_df.with_columns(
            pl.col("core_score").alias("final_phase4_score")
        )

    ce_df = normalize_ce_scores(pl.read_parquet(constants.CROSS_ENCODER_SCORES_PARQUET))

    # Left join CE scores
    merged_df = scored_df.join(ce_df, on="candidate_id", how="left")

    # Fill nulls: missing CE score falls back to core_score
    merged_df = merged_df.with_columns(
        pl.col("ce_score").fill_null(pl.col("core_score"))
    )

    # Merge: handcrafted_weight + cross_encoder_weight (from weights.yaml)
    hw = W["scoring.handcrafted_weight"]
    cew = W["scoring.cross_encoder_weight"]
    merged_df = merged_df.with_columns(
        (hw * pl.col("core_score") + cew * pl.col("ce_score")).alias("final_phase4_score")
    )

    return merged_df
