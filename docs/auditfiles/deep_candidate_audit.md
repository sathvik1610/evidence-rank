# Deep Candidate Audit — BuriBuri Top 100

**Auditor stance:** Acting as the JD author and real recruiter for Redrob AI.
**Primary source:** `candidates.jsonl` full profiles for all 100 candidates, `team_BuriBuri.csv`, `JD.txt`, `submission_spec.txt`.
**Scoring goal:** NDCG@10 (50%) + NDCG@50 (30%) are the main levers. Top-10 quality is everything.

---

## Part 1: JD Decoded — Weight Framework (Recruiter's Real Lens)

### HARD PASS/FAIL (must have ALL or disqualify)

| JD Line | What it actually means | Weight |
|---|---|---|
| *"Production experience with embeddings-based retrieval systems deployed to real users"* | Shipped BGE/E5/sentence-transformer to real users. Embedding drift, index refresh, retrieval regression in prod = the test. Not notebooks. | **Critical — #1 filter** |
| *"Production experience with vector databases or hybrid search"* | Pinecone/FAISS/Weaviate/Qdrant/Milvus/OpenSearch in production — operated it, not just called the API. | **Critical — #2 filter** |
| *"Strong Python"* | Code quality matters. Proxy: open-source, verifiable work, GitHub. | **Critical but rarely disqualifying alone** |
| *"Hands-on experience designing evaluation frameworks — NDCG, MRR, MAP, offline-to-online, A/B"* | Built an eval harness, not just called metrics.ndcg(). **The rarest skill in this dataset.** | **Critical — #3 filter, most differentiating** |

### SOFT DISQUALIFIERS (strong demotions, not hard blocks)

| JD Line | Real meaning | Weight |
|---|---|---|
| *"Pure research without production deployment"* | No academic-only PhDs. | Strong demotion |
| *"Under 12 months LangChain-only, no pre-LLM experience"* | RAG chatbot as only work = no. Longer-tenure chatbot roles still below JD bar. | Moderate demotion |
| *"Senior engineer stopped writing production code 18+ months"* | Pure arch/tech-lead without recent code. | Moderate demotion |
| *"Title-chasers: Senior → Staff → Principal every 1.5 years"* | >4 companies in 7 years with ascending titles = red flag. | Moderate demotion |
| *"Consulting firms only (TCS, Infosys, Wipro)"* | Entire career at IT services = no. Currently there but prior product = OK per JD. | Moderate demotion if no product history |
| *"CV/speech/robotics primary, without NLP/IR"* | YOLO, ASR, TTS as primary skills + no retrieval = wrong domain. | Moderate demotion |

### LOCATION WEIGHTS (recruiter reality)

| Location | JD language | Weight |
|---|---|---|
| Noida or Pune | "preferred" | Ideal, no penalty |
| Hyderabad, Mumbai, Delhi NCR | "welcome to apply" | Mild penalty |
| Any India, willing to relocate | Implied OK | Moderate penalty |
| India, not willing to relocate, outside preferred | — | Stronger penalty |
| Outside India, willing to relocate | "case-by-case" | Significant penalty (no visa sponsorship) |
| Outside India, not willing to relocate | — | Near-disqualifying for logistics |

### NOTICE PERIOD (founding team reality)

| Notice | Weight |
|---|---|
| 0 days | Ideal; slight question: why immediately available? |
| 1–15 days | JD: "love sub-30" — ideal |
| 30 days | JD: "can buy out up to 30" — ideal/buyout possible |
| 45–60 days | JD: "30+ bar gets higher" — elevated friction, not fatal |
| 90 days | Real friction; founding team = 3-month delay |
| 120 days | Near-fatal for founding team; only justified by exceptional technical profile |

### YOE (recruiter reality check)

JD says 5–9 but "some people hit senior at 4." Real weights:
- **6–8 years**: Sweet spot
- **5 years**: Fine if other signals strong
- **4 years**: OK if everything else exceptional
- **9 years**: Acceptable, slight check for "stopped coding" risk
- **>10 years**: Check for title inflation / code-stopped risk

### NICE TO HAVE (additive, not required)

- LLM fine-tuning (LoRA/QLoRA/PEFT): +bonus, especially if used in ranking context
- LTR models (XGBoost/LambdaMART): +bonus, directly relevant
- HR-tech / marketplace product context: +bonus, domain match
- Open-source contributions: +bonus, verifiable signal
- Verified Redrob assessment scores ≥80 in IR/Search/LTR: +bonus, externally verified

### BEHAVIORAL SIGNALS (Redrob platform)

| Signal | Meaning |
|---|---|
| `open_to_work=true` + response rate >70% | Actively reachable — strong positive |
| `open_to_work=false` + response rate >70% | Passive but responsive — still reachable |
| response rate <50% | Reachability concern even if strong profile |
| `last_active` >45 days ago | Losing interest or already placed |
| `saved_by_recruiters_30d` >30 | Market validation signal |
| `github_score` >70 | Good verifiable code signal |

---

## Part 2: Top-10 Deep Verdict

### Rank 1 — CAND_0046525 | Genpact AI | Pune | 6.1 yoe | 60-day notice

**Career history read:**
- **Current (48 mo, Genpact AI):** Led BM25→embedding migration across 30M+ candidate corpus. Three ranker variants in A/B. Recruiter engagement +24%, time-to-shortlist −38%. 8K peak QPS, sub-200ms p95. Focused on infra, not just modeling.
- **Prior (25 mo, LinkedIn):** 50M+ qpm ranking pipeline. BM25 + dense (BGE, FAISS HNSW). LLM reranker on top-50, LTR fallback. Offline eval from scratch — NDCG/MRR/recall@K calibrated to live A/B. ✅

**Key signals:**
- Location: Pune ✅ preferred city
- Willing to relocate: Yes ✅
- Notice: 60 days — JD says "can buy out 30" so net 30. Manageable.
- Response rate: 88% ✅
- Assessment: LangChain 96.5 (heavy LangChain emphasis — mild JD concern), Information Retrieval only 66.4 ⚠️
- Skill noise: YOLO, TTS, Redux, Image Classification — some wrong-domain noise

**Employer concern:** Genpact AI is services-branded but the actual work at Genpact AI (30M+ embedding search, A/B with recruiter metrics) is not traditional IT consulting. Prior role is LinkedIn (strong product company). JD escape clause explicitly says: "currently at these companies but prior product-company experience, that's fine."

**Verdict:** Genuine top-3. Pune preferred city + LinkedIn prior + 88% response rate. The audit calling this "not first call" overstates the concern for this specific candidate. However, ranking above CAND_0018499 (Noida, 15-day notice) is wrong on logistics. **My suggested rank: 3–4.**

---

### Rank 2 — CAND_0018499 | Zomato | Noida | 7.2 yoe | 15-day notice

**Career history read:**
- **Current (26 mo, Zomato):** 50M+ qpm, BM25+BGE+FAISS HNSW, LLM reranker on top-50, LTR fallback, offline eval from scratch — NDCG/MRR/recall@K calibrated to live A/B. ✅ ⚠️ **Same verbatim template as CAND_0046525's LinkedIn role — synthetic dataset artifact.**
- **Prior (18 mo, Google):** Same template again.
- **Prior (42 mo, Flipkart):** LLaMA-2 fine-tuning, LoRA/QLoRA, 200K preference pairs, eval harness, BentoML deployment. ✅

**Key signals:**
- Location: Noida ✅ preferred city
- Notice: 15 days ✅ ideal
- Response rate: 61% ⚠️ below 70% threshold — moderate concern
- GitHub: 94.8 ✅ very high — verifiable external signal
- Education: MIT B.Sc + NIT Surathkal M.S. ✅ top-tier
- Assessment: Deep Learning 94.0 ✅, Weaviate 72.4 ✅

**Verdict:** Should be rank 1. Preferred city + ideal notice + strong GitHub + top education + product company (Zomato). The 61% response rate is the only concern and is manageable. Description template reuse is a data quality artifact, not a candidate quality issue. **My suggested rank: 1.**

---

### Rank 3 — CAND_0064326 | Sarvam AI | Gurgaon | 7.6 yoe | 45-day notice

**Career history read:**
- **Current (31 mo, Sarvam AI):** "Owned the ranking layer for e-commerce search, hand-tuned → LTR over 9 months." Relevance labeling pipeline (click-through + human judgments), feature pipeline, training/eval workflow. "Infrastructure and data quality was the hard part." ✅ Genuine ownership statement.
- **Prior (24 mo, Aganitha):** Multiple ranking models using XGBoost/LightGBM for discovery feed. Offline-online correlation analysis. Owned PM collaboration on optimization target. ✅
- **Prior (24 mo, Freshworks):** RAG support chatbot — BLEU/ROUGE eval. Less relevant to JD.
- **Prior (12 mo, Apple):** Same LTR template as current Sarvam role. ⚠️ template reuse.

**Key signals:**
- Location: Gurgaon (Delhi NCR) ✅ welcome city
- Not willing to relocate — but Gurgaon is JD-welcome, no penalty needed
- Notice: 45 days — moderate but manageable
- Response rate: 94% ✅ excellent
- Assessment: Milvus 75.5 ✅, PyTorch 71.9

**Verdict:** Genuine top-5. Sarvam AI is a product company. LTR ownership is real (not template — has genuine infrastructure detail). Gurgaon welcome. **My suggested rank: 2–4.**

---

### Rank 4 — CAND_0046064 | Salesforce | Coimbatore | 8.9 yoe | 30-day notice

**Career history read:**
- **Current (36 mo, Salesforce):** LLaMA-2 + Mistral fine-tuning (LoRA/QLoRA) for candidate-JD matching. 200K preference pairs from recruiter labels. Eval harness (ranking metrics + human quality scores). BentoML/Kubernetes, INT8 quantization. ✅ Strong eval + fine-tuning.
- **Prior (34 mo, Verloop.io):** 50M+ qpm, BM25+BGE+FAISS, LLM reranker, LTR fallback — same template as CAND_0018499's Zomato role. ⚠️
- **Prior (36 mo, Amazon):** End-to-end ranking pipeline: BGE-large fine-tuned → Pinecone → XGBoost LTR → behavioral signals. "Hardest part was evaluation: building offline metrics that predicted live engagement." ✅ Exactly week 9–12 JD mandate.

**Key signals:**
- Location: Coimbatore — NOT in preferred or welcome list ❌
- Not willing to relocate ❌
- Notice: 30 days ✅ buyout possible
- Education: IIT Hyderabad M.Tech + BITS Pilani B.Tech ✅ top-tier
- Assessment: Python only 64.1 — surprisingly low for rank 4

**Verdict:** Strong technical profile (top-10 deserving), but Coimbatore + no relocation is a founding team problem. 30-day notice mitigates slightly. The audit calling rank 4 "slightly too high" is correct. **My suggested rank: 6–8.**

---

### Rank 5 — CAND_0002025 | Apple | Trivandrum | 5.9 yoe | 30-day notice

**Career history read:**
- **Current (42 mo, Apple):** Production recommendation system (CF + TF-IDF + sentence-transformer embeddings + behavioral re-ranking). Cold-start handling. Offline experiment → live A/B in 5 months. ✅ Shipper signal.
- **Prior (28 mo, Aganitha):** LLM fine-tuning (LoRA/QLoRA), preference pairs, eval harness — same template as CAND_0046064's Salesforce role. ⚠️

**Key signals:**
- Location: Trivandrum — NOT preferred or welcome ❌
- Not willing to relocate ❌
- Notice: 30 days ✅
- GitHub: 96.9 ✅ excellent
- Assessment: None at all ⚠️ unusual gap for top-5
- **Critical gap:** Current role is collaborative filtering + content-based recommendation. NOT hybrid search/retrieval + eval framework. The eval work is in the prior Aganitha role (LLM fine-tuning eval), not the current role. Retrieval and hybrid search evidence is thin.

**Verdict:** The audit's "too high, belongs 10-15" is correct. Strong shipper signal but current role is not the core IR/eval profile the JD requires. Trivandrum + no relocation is a real barrier. **My suggested rank: 12–18.**

---

### Rank 6 — CAND_0086022 | Sarvam AI | Kolkata | 5.3 yoe | 0-day notice

**Career history read:**
- **Current (25 mo, Sarvam AI):** Same 50M+ qpm BM25+BGE+FAISS+LLM+LTR template as CAND_0046525 (LinkedIn), CAND_0018499 (Zomato), CAND_0081846 (Razorpay). ⚠️ 4th candidate with this exact snippet at entirely different companies.
- **Prior (38 mo, Uber):** BM25→embedding migration, 30M+ corpus, A/B, recruiter engagement metrics. ✅ Product company, real context.

**Key signals:**
- Location: Kolkata — not preferred/welcome. Willing to relocate ✅
- Notice: 0 days — immediate. Why immediately available? (mild concern)
- Response rate: 55% ⚠️ below threshold
- Interview completion: 68% ⚠️ below average
- Assessment: Vector Search 92.7 ✅, pgvector 88.6 ✅, Deep Learning 79.9 ✅ — excellent verified scores
- YOE: 5.3 — low end. "Senior Applied Scientist" at 5.3 yoe — mild title concern
- Education: Stanford B.Tech claimed ✅

**Verdict:** Strong verified assessments, good Uber prior. But template description + 55% response rate + 0-day availability questions. Kolkata + relocation manageable. **My suggested rank: 8–12.**

---

### Rank 7 — CAND_0077337 | Paytm | Kochi | 7.0 yoe | 60-day notice

**Career history read:**
- **Current (19 mo, Paytm):** Recommendation system (CF + sentence-transformer + behavioral re-ranking). Same template as CAND_0002025's Apple role. ⚠️
- **Prior (14 mo, Razorpay):** Semantic search for 35M+ items. BM25→hybrid (MPNet→BGE-large fine-tuned). NDCG@10 +18%, latency −60%. ✅ The strong role.
- **Prior (44 mo, Glance):** Embedding migration template.
- **Prior (6 mo, Aganitha):** Migration template. **6 months only** ⚠️ short tenure flag.
- Career trajectory: Aganitha 6mo → Glance 44mo → Razorpay 14mo → Paytm 19mo. "Staff" title at 19-month tenure. Mild title-chaser signal per JD.

**Key signals:**
- Location: Kochi — not preferred/welcome. Willing to relocate ✅
- Notice: 60 days — elevated friction
- Response rate: 95% ✅
- Remote preferred — JD is flexible hybrid, possible friction

**Verdict:** Strong prior-role evidence (Razorpay semantic search), but current role is recommendation not IR. 6-month Aganitha stint + "Staff" title at short tenure = mild title-chaser concern. 60-day notice + Kochi. Top-10 defensible but 7 is slightly high. **My suggested rank: 9–12.**

---

### Rank 8 — CAND_0098846 | upGrad | Indore | 7.6 yoe | 45-day notice

**Career history read:**
- **Current (25 mo, upGrad):** "Owned ranking layer for e-commerce search, hand-tuned → LTR over 9 months." Same template as Sarvam AI rank 3. ⚠️
- **Prior (20 mo, Meesho):** **Churn prediction model.** MLflow, Kubeflow. This is pure MLOps/prediction — NOT search/retrieval/ranking. ⚠️
- **Prior (19 mo, Swiggy):** RAG support chatbot. Pinecone, GPT-4. BLEU/ROUGE eval. LangChain-adjacent RAG, not IR. ⚠️
- **Prior (26 mo, Google):** **Another churn prediction / MLOps role.** ⚠️
- Skill noise: YOLO, Speech Recognition, TTS — wrong domain signals
- Assessment: None at all ⚠️
- Education: PhD IIT Kanpur ✅

**Verdict:** The audit is correct — this is significantly overranked. The upGrad description is a synthetic template. The real career is churn prediction (Meesho, Google) + RAG chatbot (Swiggy). This profile does NOT have the IR/eval ownership the JD requires. Wrong-domain skill noise compounds the concern.

> **⚠️ CLEAR MISRANKING: This is the most overranked candidate in the top 10. The template inflation is the likely cause. Suggested rank: 25–38.**

---

### Rank 9 — CAND_0005538 | Adobe | Kolkata | 5.9 yoe | 90-day notice

**Career history read:**
- **Current (15 mo, Adobe):** "Led engineering team building infrastructure to surface relevant content at scale. Billions of documents, millions of queries. Index refresh, query understanding, ranking calibration, dashboards." ✅ Genuine IR infrastructure ownership.
- **Prior (30 mo, Locobuzz):** "Matching layer overhaul from hand-tuned heuristic to explicit modeling and evaluation." ✅
- **Prior (14 mo, Google):** Same description. ⚠️ template.
- Skill noise: ASR, TTS, Project Management.
- Assessment: Embeddings 92.9 ✅
- Education: SRM University B.Tech — tier 2

**Key logistics:**
- Location: Kolkata — not preferred/welcome ❌
- **Not willing to relocate** ❌
- Notice: 90 days ❌
- Double logistics failure: no relocation + 90-day notice for a founding team hire = at best remote indefinitely from Kolkata

**Verdict:** Adobe IR infra is genuinely strong. But Kolkata + no relocation + 90-day notice is a real founding team problem. JD says "30+ day notice bar gets higher." The combination makes this a rank 20–30 candidate, not top-10. **My suggested rank: 20–30.**

---

### Rank 10 — CAND_0071974 | Netflix | Vizag | 7.8 yoe | 45-day notice

**Career history read:**
- **Current (50 mo, Netflix):** "End-to-end ranking pipeline: BGE-large fine-tuned → Pinecone → XGBoost LTR → behavioral signal integration. Hardest part was evaluation: building offline metrics that predicted live engagement." ✅ The most complete technical stack in the top-10.
- **Prior (28 mo, Meta):** LLM fine-tuning, preference pairs, eval harness — template. ⚠️
- **Prior (14 mo, Mad Street Den):** Recommendation system.
- Skills: Learning to Rank, BM25, Pinecone, IR, Weaviate, Qdrant ✅ clean IR skills.
- Skill noise: Content Writing, Speech Recognition — minor noise.
- Assessment: LoRA 86.7 ✅, PEFT 85.6 ✅, Learning to Rank 77.2 ✅, Weaviate 69.2

**Key logistics:**
- Location: Vizag — not preferred/welcome ❌
- **Not willing to relocate** ❌
- Notice: 45 days — moderate

**Verdict:** Netflix end-to-end pipeline is exactly what the JD describes. This is the right profile technically. But Vizag + no relocation = founding team has to accept a permanent remote hire from a non-preferred city. "Slightly high but defensible top-15" is the correct call. **My suggested rank: 11–15.**

---

## Part 3: Key Misranked Candidates

### Severely Overranked

| Candidate | Tool Rank | Profile Reality | My Suggested Rank | Root Cause |
|---|---:|---|---:|---|
| `CAND_0098846` | 8 | Churn prediction (Meesho, Google) + RAG chatbot (Swiggy). upGrad LTR is a template. YOLO/Speech noise. No assessments. | 25–38 | Template inflation on upGrad role; real career is MLOps/chatbot |
| `CAND_0005538` | 9 | Adobe IR infra is real. 90-day notice + no relocation from Kolkata = founding team barrier. | 20–30 | Notice + no relocation penalty too weak in current system |
| `CAND_0002025` | 5 | Current role: CF/content-based recommendation, not IR/eval. Trivandrum + no relocation. | 12–18 | Recommendation vs. retrieval distinction not made in scoring |
| `CAND_0010541` | 29 | Self-describes as "production-side engineer; ML modeling work was secondary." Currently at Wysa doing fraud-detection MLOps. | 60–80 | Current role is explicit near-misalignment; honest profile undersells own weakness |
| `CAND_0065195` | 30 | Current role at CRED: RAG chatbot (Pinecone + GPT-4, BLEU/ROUGE) for 48 months. 5.1 yoe. Kolkata + no relocation. Prior: churn prediction (Google). | 55–75 | Long-tenure RAG chatbot = JD's LangChain-adjacent warning. No pre-LLM IR evidence. |
| `CAND_0037566` (LinkedIn) | 21 | Both roles (LinkedIn 51mo + Paytm 31mo) use the RAG chatbot template identically. No unique IR evidence. | 30–45 | Chatbot-only career hidden behind high response rate and 15-day notice |

### Clearly Underranked

| Candidate | Tool Rank | Why Too Low | My Suggested Rank |
|---|---:|---|---:|
| `CAND_0081846` (Razorpay) | 13 | Current role: full BM25+BGE+FAISS+LLM+LTR+eval stack. NDCG@10: 0.72→0.91 explicitly stated. 50M+ qpm. Jaipur + willing to relocate + 30-day notice. Summary is exceptional. System uses only the generic template snippet. | **4–6** |
| `CAND_0006567` (Meta) | 17 | Noida ✅ preferred city. Current: matching layer overhaul heuristic → learned relevance. Prior Razorpay (49mo): owned search/discovery/eval end-to-end. Prior Glance: offline-online A/B framework. Strong eval signal across full career. | **6–9** |
| `CAND_0080766` (Salesforce Staff) | 26 | Current (38mo, Salesforce): personalization infra, offline A/B framework, feature monitoring, drift detection, retraining cadence. 0-day notice. Willing to relocate. Swiggy prior (43mo): ranking layer + eval framework. Tier-A technical profile. not_open_to_work=false is the only real concern. | **8–14** |
| `CAND_0037980` (Niramai) | 23 | Current (38mo): "Designed ranking layer… evaluation framework that told us whether they worked… I owned all." 0-day notice. Willing to relocate. Verloop.io prior (38mo) same genuine ownership. Low GitHub (30.8) and unverified phone are concerns. | **12–18** |
| `CAND_0011687` (Niramai, rank 38) | 38 | Audit report confirmed: "owned offline-online evaluation harness — NDCG/MRR/recall calibrated to live A/B." Current job: end-to-end ranking pipeline. 15-day notice, 89% response rate. This is the week-9-12 JD mandate. CSV calls it "production depth less explicit" — contradicted by actual profile. | **15–22** |
| `CAND_0083879` (Ola) | 68 | 30-day notice, willing to relocate. Semantic search from scratch with FAISS. Ranking/eval snippets in JSONL. Ola product company. CSV says "mentions LTR with limited production context" — understated. | **35–50** |

---

## Part 4: Template Reuse / Description Collapse — The Most Important Finding

**This is the single most important data quality insight from reading the full profiles.**

The following career description templates appear **verbatim or near-verbatim across multiple candidates at entirely different companies:**

| Template Text | Appears in candidates at |
|---|---|
| *"50M+ qpm, BM25+BGE+FAISS HNSW, LLM reranker on top-50, LTR fallback, offline eval from scratch — NDCG/MRR/recall@K calibrated to live A/B"* | Genpact AI, Zomato, Google, Sarvam AI, Razorpay — at least 8 candidates |
| *"30M+ corpus, BM25→embedding migration, 3 ranker variants, A/B, recruiter engagement +24%, time-to-shortlist −38%"* | Genpact AI (current), Uber, Glance |
| *"Owned ranking layer from hand-tuned → LTR over 9 months, relevance labeling (click-through + human judgments), feature pipeline, training/eval"* | Sarvam AI (rank 3 current), upGrad (rank 8 current), Haptik (rank 27), CRED (rank 19) |
| *"RAG chatbot, Pinecone, GPT-4, fine-tuned smaller model, BLEU/ROUGE, human-in-the-loop"* | Amazon (rank 11), LinkedIn (rank 21), upGrad (rank 16 prior), CRED (rank 30), Mad Street Den (rank 19) |
| *"Recommendation system 10M+ users, CF + sentence-transformer embeddings, GB model, A/B"* | Genpact AI (prior), Microsoft (current), Freshworks (current) |

**Why this matters:** The cross-encoder and BM25 retrieval are treating these templates as strong evidence for every candidate that has them — regardless of whether the rest of that candidate's career supports it. This inflates scores for candidates like CAND_0098846 (rank 8) whose only IR evidence is the template while the rest of their career is churn prediction and chatbots.

**The fix:** Do NOT just weight "current role" higher than "prior role"—in a synthetic dataset, ground truth likely scores the whole profile. The better framing is **Career IR Density**: "Does the *majority* of this candidate's career have IR/ranking/eval evidence, or is IR evidence isolated to one template in one role while the rest is churn prediction/MLOps?" The system should weight template evidence proportionally less if the candidate's career pattern (majority of roles) does not support it.

---

## Part 5: Honeypot Analysis

**No honeypots confirmed in top 100.** Key verification:
- No candidate flagged `impossible_flag`, `suspicious_flag`, `is_ghost`, `consulting_only`, or hard `wrong_domain`
- All YOE values plausible given career histories
- No "expert in 10 skills with 0 years used" pattern observed
- Template reuse is a **synthetic dataset artifact**, not honeypot behavior

**High-risk candidates (suspicious quality, not honeypot):**
- `CAND_0052328` (Amazon, rank 11): RAG chatbot as current role for 52 months. Unverified email AND phone. ⚠️
- `CAND_0010541` (Wysa, rank 29): Explicitly self-describes as secondary ML engineer on someone else's DS work. ⚠️
- `CAND_0098846` (upGrad, rank 8): Churn prediction + chatbot career with template LTR description. No assessments. ⚠️

**Conclusion:** 0% honeypot rate in top 100. System correctly avoids honeypots. ✅

---

## Part 6: My Recruiter Top-10

| Rank | Candidate | Company | Location | Notice | Key Reason |
|---:|---|---|---|---|---|
| 1 | `CAND_0018499` | Zomato | Noida ✅ | 15d ✅ | Preferred city + ideal notice + 50M+ qpm search + eval + GitHub 94.8 |
| 2 | `CAND_0081846` | Razorpay | Jaipur→reloc ✅ | 30d ✅ | Full BM25+dense+LLM+LTR+eval stack, NDCG 0.72→0.91 explicit, product company |
| 3 | `CAND_0064326` | Sarvam AI | Gurgaon ✅ | 45d | NCR welcome city, LTR ownership from scratch, relevance labeling, 94% response |
| 4 | `CAND_0046525` | Genpact AI | Pune ✅ | 60d | Preferred city, LinkedIn prior (full stack, 50M+ qpm), 88% response rate |
| 5 | `CAND_0006567` | Meta | Noida ✅ | 60d | Preferred city, matching layer overhaul, Razorpay prior 49mo (search/discovery/eval) |
| 6 | `CAND_0046064` | Salesforce | Coimbatore | 30d | IIT+BITS education, LLM eval harness + Amazon end-to-end pipeline. No relocation = friction but buyout possible |
| 7 | `CAND_0080766` | Salesforce Staff | Coimbatore→reloc ✅ | 0d | Immediate, relocating, Salesforce personalization + Swiggy ranking-layer ownership. not_open concern partially offset by 0-day |
| 8 | `CAND_0086022` | Sarvam AI | Kolkata→reloc ✅ | 0d | Strong verified assessments (Vector Search 92.7, pgvector 88.6), Uber prior (30M+ migration). 55% response rate is main concern |
| 9 | `CAND_0077337` | Paytm | Kochi→reloc ✅ | 60d | Razorpay prior (35M+ semantic search, NDCG +18%, latency −60%), 95% response rate |
| 10 | `CAND_0027691` | Haptik | Pune ✅ | 15d | Preferred city + ideal notice + Vedantu prior semantic search + Weaviate 79.3 assessment |

**Displaced:** CAND_0071974 (Netflix, strong tech, but Vizag + no relocation = founding team barrier) → rank 11–14. CAND_0098846 (upGrad) → rank 25–38. CAND_0005538 (Adobe) → rank 20–30. CAND_0002025 (Apple) → rank 12–18.

---

## Part 7: Top-50 Out-of-Bounds

**Should not be in top-50:**

| Candidate | Tool Rank | Reason |
|---|---:|---|
| `CAND_0098846` | 8 | Churn prediction + chatbot career |
| `CAND_0010541` | 29 | Self-described secondary engineer |
| `CAND_0065195` | 30 | Long-tenure RAG chatbot, no prior IR |
| `CAND_0037566` (LinkedIn) | 21 | Both roles are RAG chatbot template |

**Should be in top-50 but ranked below it:**

| Candidate | Tool Rank | Reason |
|---|---:|---|
| `CAND_0011687` | 38 | Eval harness ownership explicitly stated; 15-day notice, 89% response — belongs top-20 |
| `CAND_0083879` | 68 | Semantic search from scratch, FAISS, 30-day notice, Ola product company |
| `CAND_0060054` | 83 | Jaipur + relocation + 15-day notice + 86% response + semantic search + FAISS/BGE |

---

## Part 8: Scoring Impact Assessment — Correct Understanding of NDCG

### How the scoring actually works (submission_spec.txt)

| Metric | Weight | What it actually measures |
|---|---|---|
| NDCG@10 | 50% | Relevance-weighted quality of your top-10 SET |
| NDCG@50 | 30% | Relevance-weighted quality of your top-50 SET |
| MAP | 15% | Precision across all relevance levels |
| P@10 | 5% | Fraction of your top-10 that are "relevant" (tier 3+) — pure SET MEMBERSHIP |

### The critical insight: SET MEMBERSHIP > exact ordering

NDCG@K formula: `DCG@K = sum(relevance_i / log2(i+1))` for i=1..K.

The key reality:
- The gain from moving a candidate from rank 11 → rank 1 (IN vs OUT of top-10) is **massive**
- The gain from moving rank 1 → rank 2 within the top-10 **among equally strong candidates** is small
- P@10 is pure set membership — zero position sensitivity
- Honeypot tolerance is **≤10%** (not 0%) precisely because the spec authors know no system perfectly matches ground truth order

**The spec does NOT expect exact order match. They expect the right CANDIDATES to be in the right WINDOWS (top-10, top-50).**

---

### Priority 1 — Wrong candidates IN top-10 (highest score damage)

These candidates are consuming top-10 slots that should belong to genuinely strong candidates. Each one at rank ≤10 directly costs NDCG@10 (50% of total) and P@10:

| Candidate | Tool Rank | Why they should NOT be in top-10 | Score damage |
|---|---:|---|---|
| `CAND_0098846` | 8 | Real career: churn prediction (Meesho, Google) + RAG chatbot (Swiggy). LTR description is template inflation. If ground truth puts this at tier 1-2, it's costing massive NDCG@10. | **Highest priority fix** |
| `CAND_0005538` | 9 | Adobe IR is real but 90-day notice + no relocation from Kolkata. Ground truth almost certainly demotes logistically near-impossible candidates for a founding team hire. | **High priority fix** |
| `CAND_0002025` | 5 | Current role is recommendation (CF), not IR/eval. Trivandrum + no relocation. Consuming rank 5 slot. | **High priority fix** |

### Priority 2 — Right candidates OUTSIDE top-10 when they should be IN it

Every ground-truth top-10 candidate that appears at rank 11–20 instead of rank 1–10 is a missed NDCG@10 contribution:

| Candidate | Tool Rank | Why they SHOULD be in top-10 | Score gain from fixing |
|---|---:|---|---|
| `CAND_0081846` | 13 | Full BM25+dense+LLM+LTR+eval stack. NDCG 0.72→0.91 explicit. Razorpay product company. 30-day notice, willing to relocate. If ground truth has this in top-10, missing it costs hugely. | **Highest priority fix** |
| `CAND_0006567` | 17 | Noida preferred city. Strong eval signal across full career (Razorpay 49mo, Glance A/B framework). If ground truth has this top-10, rank 17 is a major miss. | **High priority fix** |

### Priority 3 — Wrong candidates IN top-50 (NDCG@50 damage)

| Candidate | Tool Rank | Should be at |
|---|---:|---|
| `CAND_0010541` | 29 | 60–80 — self-described secondary engineer |
| `CAND_0065195` | 30 | 55–75 — RAG chatbot only, no pre-LLM IR |
| `CAND_0037566` | 21 | 30–45 — both roles are RAG chatbot template |

### Priority 4 — Right candidates OUTSIDE top-50 (NDCG@50 missed)

| Candidate | Tool Rank | Should be at |
|---|---:|---|
| `CAND_0011687` | 38 | 15–22 — explicit eval harness ownership, 15-day notice, 89% response |
| `CAND_0083879` | 68 | 35–50 — semantic search from scratch, Ola product company |
| `CAND_0060054` | 83 | 45–60 — Jaipur + relocation + 15-day notice |

### Priority 5 — Ordering swaps WITHIN top-10 (lowest impact)

Whether `CAND_0018499` is rank 1 vs rank 2, or `CAND_0046525` is rank 3 vs rank 4 — **if both are genuinely top-10 quality** — this is the smallest scoring lever. The DCG gain of rank 1 vs rank 2 is `1.0 vs 0.63`. Important but small compared to the binary membership question.

**Bottom line:** Fix who is IN the windows first. Fix their internal order second.

---

*Audit completed by Antigravity. All judgments are based on full profile data from candidates.jsonl, team_BuriBuri.csv, JD.txt (line-by-line weight framework), and submission_spec.txt scoring criteria. No code was changed.*

