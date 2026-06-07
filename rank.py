"""
rank.py — Stage B Runtime Entry Point

Lightning-fast runtime orchestrator.
Executes Phase 4 (Core Scoring), Phase 5 (Behavioral), and Phase 6 (Explanations).
Relies entirely on precomputed artifacts to stay well under the 5-minute constraint.

No imports of torch, faiss, or FlagEmbedding.
"""

import sys
import argparse
import polars as pl
import pandas as pd
from datetime import date
import time
import json
import os

import constants
from src.scorer import score_candidates_vectorized
from src.reranker import merge_cross_encoder_scores
from src.behavioral import compute_final_score, assign_ranks
from src.explainer import generate_reasoning, get_largest_concern

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates JSONL")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    print("--- Stage B: Ranking Execution ---")
    
    # 1. Load Precomputed Features (Phase 1c output)
    # The offline pipeline already filtered this to the Top 5000 from Phase 1d (RRF)
    if not os.path.exists(constants.CANDIDATE_FEATURES_PARQUET):
        print(f"Error: Precomputed features not found at {constants.CANDIDATE_FEATURES_PARQUET}.")
        print("Run preprocess.py first.")
        sys.exit(1)
        
    df = pl.read_parquet(constants.CANDIDATE_FEATURES_PARQUET)
    print(f"Loaded {len(df)} candidates from precomputed feature pool.")

    # 2. Phase 4a: Core Scoring (Vectorized)
    df = score_candidates_vectorized(df)
    
    # 3. Slice to Top 500
    df = df.sort("core_score", descending=True).head(500)
    print(f"Sliced to top {len(df)} candidates by core score.")
    
    # 4. Phase 4b: Cross-Encoder Merge
    df = merge_cross_encoder_scores(df)
    
    # We now move to python dictionaries for Phase 5 & 6 row-level operations
    # since these are complex heuristic formulas and we only have 500 records.
    candidates = df.to_dicts()
    
    # Reference date for ghost detection and reachability
    # We load run metadata to get the actual reference time, or default to mid-2026
    ref_date = date(2026, 6, 1)
    if os.path.exists(constants.RUN_METADATA_JSON):
        with open(constants.RUN_METADATA_JSON, "r") as f:
            meta = json.load(f)
            if "run_time" in meta:
                ref_date = date.fromisoformat(meta["run_time"].split("T")[0])

    print("Running Phase 5 (Behavioral Modifiers)...")
    for cand in candidates:
        cand["final_score"] = compute_final_score(cand, ref_date)
        
    # Phase 5b: Rank Assignment
    candidates = assign_ranks(candidates)
    
    # Drop candidates with 0.0 final_score (ghosts / fully penalized)
    # The hackathon expects Top 100 in the CSV. We'll generate reasons for all >0 
    # but only output the top 100.
    valid_cands = [c for c in candidates if c["final_score"] > 0.0]
    top_100 = valid_cands[:100]
    
    print(f"Running Phase 6 (Reason Generation) for Top {len(top_100)}...")
    debug_records = []
    output_records = []
    
    for cand in top_100:
        reason = generate_reasoning(cand)
        
        output_records.append({
            "candidate_id": cand["candidate_id"],
            "rank": cand["rank"],
            "score": cand["final_score"],
            "reasoning": reason
        })
        
        # Debug trace
        debug_records.append({
            "candidate_id": cand["candidate_id"],
            "rank": cand["rank"],
            "score": cand["final_score"],
            "core_score": round(cand.get("core_score", 0.0), 2),
            "ce_score": round(cand.get("ce_score", 0.0), 2),
            "reasoning": reason,
            "concern": get_largest_concern(cand)
        })
        
    # Save Outputs
    out_df = pd.DataFrame(output_records)
    out_df.to_csv(args.out, index=False)
    
    # Save Debug (Offline only)
    debug_df = pd.DataFrame(debug_records)
    debug_df.to_csv("artifacts/ranking_debug.csv", index=False)
    
    elapsed = time.time() - start_time
    print(f"Successfully wrote {len(out_df)} candidates to {args.out}")
    print(f"Debug trace written to artifacts/ranking_debug.csv")
    print(f"Runtime: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
