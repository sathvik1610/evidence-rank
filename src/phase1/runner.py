import json
import os
import pandas as pd
from typing import Dict, Any

from .date_utils import compute_chrono_months
from .hard_rules import check_hard_eliminations
from .soft_signals import extract_soft_signals, check_ghost

def execute(input_file: str = r"d:\GitHub\evidence-rank\Resources\candidates.jsonl"):
    print("Starting Phase 1 Execution...")
    
    os.makedirs("artifacts", exist_ok=True)
    
    eliminated_file = open("artifacts/phase1_eliminated.jsonl", "w", encoding="utf-8")
    survivors_file = open("artifacts/phase1_survivors.jsonl", "w", encoding="utf-8")
    
    flags_data = []
    total = 0
    eliminated_count = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if total % 10000 == 0:
                print(f"Phase 1: Processed {total} candidates...")
                
            candidate = json.loads(line)
            cid = candidate["candidate_id"]
            
            chrono_months = compute_chrono_months(candidate.get("career_history", []))
            is_hp, hp_reason = check_hard_eliminations(candidate, chrono_months)
            
            if is_hp:
                eliminated_count += 1
                candidate["_elimination_reason"] = hp_reason
                eliminated_file.write(json.dumps(candidate) + "\n")
                
                flags_data.append({
                    "candidate_id": cid,
                    "is_honeypot": True,
                    "is_ghost": False,
                    "product_ratio": 0.0,
                    "consulting_only": False,
                    "research_only": False,
                    "wrong_domain": False,
                    "contradiction_skill_duration": 0,
                    "contradiction_assessment": 0
                })
                continue
            
            # If not honeypot, calculate soft signals
            survivors_file.write(json.dumps(candidate) + "\n")
            
            is_ghost = check_ghost(candidate)
            soft_signals = extract_soft_signals(candidate, chrono_months)
            
            flags = {
                "candidate_id": cid,
                "is_honeypot": False,
                "is_ghost": is_ghost
            }
            flags.update(soft_signals)
            flags_data.append(flags)
            
    eliminated_file.close()
    survivors_file.close()

    df = pd.DataFrame(flags_data)
    df.to_parquet("artifacts/candidate_flags.parquet")
    
    print(f"Phase 1 complete! Total: {total}, Eliminated: {eliminated_count}, Survivors: {total - eliminated_count}")
    print("Saved: artifacts/candidate_flags.parquet")
    print("Saved: artifacts/phase1_eliminated.jsonl")
    print("Saved: artifacts/phase1_survivors.jsonl")

if __name__ == "__main__":
    execute()
