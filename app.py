"""
app.py — HuggingFace Spaces Gradio Demo + CLI Runner

Runs the heuristic ranking pipeline without any heavy ML models or artifacts.
Phases executed: 1f (Honeypot), 1c (Features), 4 (Score), 5 (Behavioral), 6 (Reasoning)

Usage (CLI):
    python app.py --candidates ./sample_candidates.json
    python app.py --candidates ./candidates.jsonl --out ./output.csv

Usage (Gradio sandbox):
    python app.py
"""

import argparse
import json
import sys
import csv
import gzip
from datetime import date
import pandas as pd

from src.features import extract_features, compute_product_ratio
from preprocess import _check_impossible_flag, _compute_honeypot_score, _is_ghost
from src.scorer import compute_core_score
from src.behavioral import compute_final_score, assign_ranks
from src.explainer import generate_reasoning


def load_candidates(path: str) -> list:
    """Load candidates from .json, .jsonl, or .jsonl.gz files."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) == 1:
            return json.loads(lines[0])
        return [json.loads(l) for l in lines if l.strip()]
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSONL first (multiple lines of JSON objects)
    if content.startswith("{"):
        return [json.loads(l) for l in content.splitlines() if l.strip()]
    
    # Otherwise treat as a JSON array
    return json.loads(content)


def rank_candidates(candidates: list) -> list:
    """Run the full heuristic pipeline on a list of candidate dicts."""
    ref_date = date.today()
    scored = []

    for c in candidates:
        impossible = _check_impossible_flag(c)
        hp_score = _compute_honeypot_score(c)
        is_ghost_flag = _is_ghost(c, ref_date)
        product_ratio = compute_product_ratio(c)

        cf = {
            "impossible_flag": impossible,
            "honeypot_score": hp_score,
            "suspicious_flag": hp_score > 0.70,
            "is_ghost": is_ghost_flag,
            "product_ratio": product_ratio,
            "consulting_only": product_ratio == 0.0,
            "research_only": False,
            "wrong_domain": False,
        }

        flat = extract_features(c, cf)
        core_score = compute_core_score(flat)
        flat["core_score"] = core_score
        flat["final_phase4_score"] = core_score
        flat["final_score"] = compute_final_score(flat, ref_date)
        scored.append(flat)

    return assign_ranks(scored)


def write_csv(ranked: list, out_path: str):
    """Write ranked candidates to a submission-format CSV."""
    rows = [
        {
            "candidate_id": c.get("candidate_id", "UNKNOWN"),
            "rank": c["rank"],
            "score": round(c["final_score"], 6),
            "reasoning": generate_reasoning(c),
        }
        for c in ranked
        if c["final_score"] > 0
    ]
    rows.sort(key=lambda x: x["rank"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Ranked {len(rows)} candidates -> {out_path}")


# ─── CLI mode ────────────────────────────────────────────────────────────────

def run_cli():
    parser = argparse.ArgumentParser(
        description="Evidence Rank — heuristic candidate ranker (no artifacts required)"
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates file (.json, .jsonl, or .jsonl.gz)"
    )
    parser.add_argument(
        "--out", default="app_output.csv",
        help="Output CSV path (default: app_output.csv)"
    )
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.")

    print("Running heuristic pipeline (no GPU / no artifacts needed)...")
    ranked = rank_candidates(candidates)
    write_csv(ranked, args.out)


# ─── Gradio sandbox mode ─────────────────────────────────────────────────────

def rank_demo(json_text: str) -> pd.DataFrame:
    try:
        candidates = json.loads(json_text)
        if not isinstance(candidates, list):
            return pd.DataFrame({"Error": ["Payload must be a list of candidate JSON objects."]})
    except Exception as e:
        return pd.DataFrame({"Error": [f"Invalid JSON: {str(e)}"]})

    if not candidates:
        return pd.DataFrame()

    ranked = rank_candidates(candidates)
    results = [
        {
            "Rank": c["rank"],
            "Candidate ID": c.get("candidate_id", "UNKNOWN"),
            "Score": round(c["final_score"], 4),
            "Reason": generate_reasoning(c),
        }
        for c in ranked
        if c["final_score"] > 0
    ]

    if not results:
        return pd.DataFrame({"Message": ["No candidates passed the filtering threshold."]})

    return pd.DataFrame(results)


def rank_demo_file(file) -> pd.DataFrame:
    """Handle file upload in Gradio sandbox."""
    if file is None:
        return pd.DataFrame({"Error": ["No file uploaded."]})
    try:
        candidates = load_candidates(file.name)
        if len(candidates) > 100:
            candidates = candidates[:100]
        ranked = rank_candidates(candidates)
        results = [
            {
                "Rank": c["rank"],
                "Candidate ID": c.get("candidate_id", "UNKNOWN"),
                "Score": round(c["final_score"], 4),
                "Reason": generate_reasoning(c),
            }
            for c in ranked
            if c["final_score"] > 0
        ]
        if not results:
            return pd.DataFrame({"Message": ["No candidates passed the filtering threshold."]})
        return pd.DataFrame(results)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


def launch_gradio():
    import gradio as gr

    with gr.Blocks(title="Evidence Rank - Candidate Scoring Sandbox") as demo:
        gr.Markdown("# Evidence Rank Sandbox")
        gr.Markdown(
            "Rank up to 100 candidates instantly. No GPU or artifacts folder required. "
            "Uses heuristic pipeline: Phases 1f (Honeypot), 1c (Features), 4 (Score), 5 (Behavioral), 6 (Reasoning)."
        )
        with gr.Tabs():
            with gr.Tab("Upload File"):
                gr.Markdown("Upload a `.json` or `.jsonl` candidates file (max 100 candidates).")
                file_input = gr.File(
                    label="Upload candidates (.json / .jsonl)",
                    file_types=[".json", ".jsonl"]
                )
                file_btn = gr.Button("Rank Uploaded File", variant="primary")
                file_output = gr.Dataframe(headers=["Rank", "Candidate ID", "Score", "Reason"])
                file_btn.click(fn=rank_demo_file, inputs=file_input, outputs=file_output)

            with gr.Tab("Paste JSON"):
                gr.Markdown("Paste a JSON array of candidate objects directly.")
                input_json = gr.Textbox(
                    lines=15,
                    placeholder='[{"candidate_id": "CAND_0000001", "profile": {...}}]',
                    label="Input Candidates (JSON List)",
                )
                paste_btn = gr.Button("Rank Candidates", variant="primary")
                paste_output = gr.Dataframe(headers=["Rank", "Candidate ID", "Score", "Reason"])
                paste_btn.click(fn=rank_demo, inputs=input_json, outputs=paste_output)

    demo.launch()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--candidates" in sys.argv:
        run_cli()
    else:
        launch_gradio()
