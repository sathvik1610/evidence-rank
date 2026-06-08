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
from src.weights import W


def load_candidate_ids(path: str) -> set[str]:
    """
    Read only candidate_id values from a JSON array or JSONL file.
    This guards against stale artifacts being ranked for the wrong input file.
    """
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            for item in data:
                if isinstance(item, dict) and item.get("candidate_id"):
                    ids.add(str(item["candidate_id"]))
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict) and item.get("candidate_id"):
                    ids.add(str(item["candidate_id"]))
    return ids

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
        
    candidate_ids = load_candidate_ids(args.candidates)
    if not candidate_ids:
        print(f"Error: no candidate_id values found in {args.candidates}.")
        sys.exit(1)

    df = pl.read_parquet(constants.CANDIDATE_FEATURES_PARQUET)
    before_filter = len(df)
    df = df.filter(pl.col("candidate_id").is_in(candidate_ids))
    print(f"Loaded {len(df)} candidates from precomputed feature pool after filtering {before_filter} artifact rows to the input file.")
    if len(df) == 0:
        print("Error: input candidate IDs do not overlap with precomputed features. Re-run preprocess.py for this candidate file.")
        sys.exit(1)

    # Phase 2: if RRF retrieval scores are available, honor the runtime retrieval
    # cutoff before core scoring. Sample --skip-embed runs do not create this file,
    # so they correctly exercise all sample candidates.
    if os.path.exists(constants.RETRIEVAL_SCORES_PARQUET):
        retrieval_df = pl.read_parquet(constants.RETRIEVAL_SCORES_PARQUET)
        runtime_top_k = int(W.get("retrieval.runtime_top_k", constants.RUNTIME_RETRIEVAL_TOPK))
        df = (
            df.join(retrieval_df, on="candidate_id", how="left")
              .with_columns(pl.col("rrf_score").fill_null(0.0))
              .sort("rrf_score", descending=True)
              .head(runtime_top_k)
        )
        print(f"Applied Phase 2 RRF cutoff: top {len(df)} by retrieval score.")

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
            # GAP 3 FIX: use reference_date (= max last_active_date from corpus)
            # Fall back to run_time for backwards compatibility with old artifacts
            date_str = meta.get("reference_date") or (
                meta.get("run_time", "").split("T")[0] if meta.get("run_time") else None
            )
            if date_str:
                try:
                    ref_date = date.fromisoformat(date_str)
                except ValueError:
                    pass

    print("Running Phase 5 (Behavioral Modifiers)...")
    for cand in candidates:
        cand["final_score"] = compute_final_score(cand, ref_date)
        
    # Phase 5b: Rank Assignment
    candidates = assign_ranks(candidates)
    
    # Official submissions must contain exactly 100 rows. For sample/sandbox
    # inputs with fewer than 100 candidates, output the available ranked rows.
    if len(candidate_ids) >= 100:
        top_100 = candidates[:100]
    else:
        top_100 = candidates
    
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
