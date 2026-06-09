# Intelligent Candidate Discovery & Ranking Engine
## Production System Specification — Version 3.3.0

**Project:** Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge  
**Role Being Ranked For:** Senior AI Engineer, Founding Team, Redrob AI  
**Dataset:** 100,000 candidates in `candidates.jsonl` or `candidates.jsonl.gz`  
**Output:** `team_BuriBuri.csv` — top 100 candidates, ranked best-fit first  
**Hard Constraints:** 5-minute wall-clock execution (ranking step only), CPU-only, 16 GB RAM, no network calls during ranking

---

## Table of Contents

[1. Problem Understanding & Design Philosophy](./common_architecture.md)  
[2. System Architecture Overview](./common_architecture.md)  
[2a. Current Architecture Guide](../ARCHITECTURE.md)  
[2b. Judge Guide](../JUDGE_GUIDE.md)  
[2c. Data And Artifacts](../DATA_AND_ARTIFACTS.md)  
[2d. Reproducibility Guide](../../REPRODUCIBILITY.md)  
[2e. Execution Model](./common_architecture.md)  
[3. Repository Layout](./common_architecture.md)  
[4. Phase 0 — JD Intelligence](./phase_0_jd_intelligence.md)  
[5. Phase 1 — Corpus Preprocessing + Honeypot Detection](./phase_1_corpus_preprocessing.md)  
[6. Phase 2 — Multi-Signal Retrieval](./phase_2_multi_signal_retrieval.md)  
[7. Phase 3 — Candidate Feature Extraction](./phase_3_feature_extraction.md)  
[8. Phase 4 — Core Scoring + Cross-Encoder Rerank](./phase_4_core_scoring.md)  
[9. Phase 5 — Behavioral Re-ranking + Penalization](./phase_5_behavioral_reranking.md)  
[10. Phase 6 — Reason Generation](./phase_6_reason_generation.md)  
[11. Phase 7 — Manual Validation + Weight Tuning](./phase_7_validation_and_references.md)  
[12. Data Field Reference](./phase_7_validation_and_references.md)  
[13. Model & Library Choices](./phase_7_validation_and_references.md)  
[14. Dependency Declarations](./phase_7_validation_and_references.md)

---

