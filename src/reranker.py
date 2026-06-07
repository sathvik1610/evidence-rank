"""
src/reranker.py — Phase 4: Cross-Encoder Merge (Runtime)

Loads the offline-computed bge-reranker-v2-m3 scores and merges them with the
handcrafted core_score to produce final_phase4_score.
"""

import polars as pl
import os
import constants

def merge_cross_encoder_scores(scored_df: pl.DataFrame) -> pl.DataFrame:
    """
    Left-joins the precomputed cross-encoder scores and applies the 80/20 merge.
    Gracefully falls back to core_score if CE score is missing (or if we are running without CE).
    """
    if not os.path.exists(constants.CROSS_ENCODER_SCORES_PARQUET):
        # Graceful degradation if CE parquet doesn't exist (e.g. testing)
        print(f"Warning: {constants.CROSS_ENCODER_SCORES_PARQUET} not found. Skipping CE merge.")
        return scored_df.with_columns(
            pl.col("core_score").alias("final_phase4_score")
        )
        
    ce_df = pl.read_parquet(constants.CROSS_ENCODER_SCORES_PARQUET)
    
    # Left join CE scores
    merged_df = scored_df.join(ce_df, on="candidate_id", how="left")
    
    # Fill nulls with core_score (handles candidates missing from CE pool)
    merged_df = merged_df.with_columns(
        pl.col("ce_score").fill_null(pl.col("core_score"))
    )
    
    # Merge: 80% handcrafted, 20% cross-encoder
    merged_df = merged_df.with_columns(
        (0.8 * pl.col("core_score") + 0.2 * pl.col("ce_score")).alias("final_phase4_score")
    )
    
    return merged_df
