import csv
import json
import os
import sys

def main():
    disqualified_csv = os.path.join("artifacts", "hard_disqualified_debug.csv")
    candidates_jsonl = "candidates.jsonl"
    output_csv = os.path.join("artifacts", "top_10_disqualified_profiles.csv")
    
    # 1. Read top 10 disqualified candidate IDs
    if not os.path.exists(disqualified_csv):
        print(f"Error: {disqualified_csv} not found.")
        sys.exit(1)
        
    candidate_ids = []
    with open(disqualified_csv, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("Error: Empty CSV file.")
            sys.exit(1)
        for row in reader:
            if row:
                candidate_ids.append(row[0])
                if len(candidate_ids) == 10:
                    break
                    
    if not candidate_ids:
        print("No candidates found in disqualified CSV.")
        sys.exit(0)
        
    print(f"Top 10 disqualified candidates to extract: {candidate_ids}")
    
    # 2. Scan candidates.jsonl and retrieve profiles
    if not os.path.exists(candidates_jsonl):
        print(f"Error: {candidates_jsonl} not found.")
        sys.exit(1)
        
    profiles_map = {}
    with open(candidates_jsonl, mode="r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cand = json.loads(line)
            cid = cand.get("candidate_id")
            if cid in candidate_ids:
                profiles_map[cid] = cand
                if len(profiles_map) == len(candidate_ids):
                    break # Found all of them
                    
    # 3. Write profiles in the order they appeared in the disqualified list
    headers = [
        "candidate_id",
        "anonymized_name",
        "headline",
        "years_of_experience",
        "current_title",
        "current_company",
        "location",
        "country",
        "summary",
        "skills",
        "education",
        "career_history",
        "certifications",
        "languages",
        "profile_completeness_score",
        "open_to_work_flag",
        "github_activity_score"
    ]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for cid in candidate_ids:
            cand = profiles_map.get(cid)
            if not cand:
                print(f"Warning: Candidate {cid} not found in JSONL.")
                continue
                
            prof = cand.get("profile", {})
            signals = cand.get("redrob_signals", {})
            
            # Format skills
            skills_formatted = "; ".join([
                f"{s.get('name', '')} ({s.get('proficiency', '')}, {s.get('duration_months', 0)} mos)"
                for s in cand.get("skills", [])
            ])
            
            # Format education
            edu_formatted = "; ".join([
                f"{e.get('degree', '')} in {e.get('field_of_study', '')} from {e.get('institution', '')} ({e.get('start_year', '')}-{e.get('end_year', '')}, Grade: {e.get('grade', '')})"
                for e in cand.get("education", [])
            ])
            
            # Format career history
            career_formatted = "; ".join([
                f"{c.get('title', '')} at {c.get('company', '')} ({c.get('start_date', '')} to {c.get('end_date') or 'Present'}, {c.get('duration_months', 0)} mos) - {c.get('description', '')}"
                for c in cand.get("career_history", [])
            ])
            
            # Format certifications
            cert_formatted = "; ".join([
                f"{cert.get('name', '')} by {cert.get('issuer', '')} ({cert.get('year', '')})"
                for cert in cand.get("certifications", [])
            ])
            
            # Format languages
            lang_formatted = "; ".join([
                f"{l.get('language', '')} ({l.get('proficiency', '')})"
                for l in cand.get("languages", [])
            ])
            
            row = [
                cid,
                prof.get("anonymized_name", ""),
                prof.get("headline", ""),
                prof.get("years_of_experience", ""),
                prof.get("current_title", ""),
                prof.get("current_company", ""),
                prof.get("location", ""),
                prof.get("country", ""),
                prof.get("summary", ""),
                skills_formatted,
                edu_formatted,
                career_formatted,
                cert_formatted,
                lang_formatted,
                signals.get("profile_completeness_score", ""),
                signals.get("open_to_work_flag", ""),
                signals.get("github_activity_score", "")
            ]
            writer.writerow(row)
            
    print(f"Successfully wrote top 10 disqualified profiles to {output_csv}")

if __name__ == "__main__":
    main()
