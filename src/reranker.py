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

    ce_df = pl.read_parquet(constants.CROSS_ENCODER_SCORES_PARQUET)

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
