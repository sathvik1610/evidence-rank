import sys
sys.path.insert(0, ".")

from src.runtime_calibration import calibrate_candidate


def test_python_proxy_from_mlflow_kubeflow_tooling():
    cand = {}
    profile = {
        "profile": {
            "current_title": "Recommendation Systems Engineer",
            "headline": "Production recommendation systems engineer",
            "summary": "Builds ranking systems.",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "title": "Recommendation Systems Engineer",
                "company": "Microsoft",
                "industry": "Software",
                "description": (
                    "Built and operated production ML pipelines using MLflow "
                    "for experiment tracking and Kubeflow for orchestration."
                ),
            }
        ],
        "skills": [{"name": "FAISS"}],
    }

    calibrated = calibrate_candidate(cand, profile)
    assert calibrated["runtime_career_python_signal"] == 1.0


def test_faiss_python_proxy_requires_hands_on_ml_shipping_context():
    cand = {}
    profile = {
        "profile": {
            "current_title": "Search Engineer",
            "headline": "Semantic search engineer",
            "summary": "Ships search systems.",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "title": "Search Engineer",
                "company": "ProductCo",
                "industry": "Software",
                "description": (
                    "Shipped semantic search using sentence-transformers and FAISS "
                    "with monitoring and rollback paths."
                ),
            }
        ],
        "skills": [],
    }

    calibrated = calibrate_candidate(cand, profile)
    assert calibrated["runtime_career_python_signal"] == 1.0


def test_bare_vector_db_skill_does_not_prove_python_gate():
    cand = {}
    profile = {
        "profile": {
            "current_title": "Product Manager",
            "headline": "AI product manager",
            "summary": "Worked with vector search products.",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "title": "Product Manager",
                "company": "ProductCo",
                "industry": "Software",
                "description": "Coordinated Pinecone rollout with engineering teams.",
            }
        ],
        "skills": [{"name": "Pinecone"}, {"name": "Qdrant"}],
    }

    calibrated = calibrate_candidate(cand, profile)
    assert calibrated["runtime_career_python_signal"] == 0.0
