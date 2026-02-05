import sqlalchemy as sa
from utils.db_manager import get_engine
from utils.llm_client import run_llm
import json


class AutoMLAgent:
    """
    Uses LLM to recommend ML models based on engineered features.
    Stores results in model_recommendations table.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.engine = get_engine()

    def run(self):
        # 1️ Prompt for LLM
        prompt = """
        You are an expert AutoML system.
        Given engineered tabular features, recommend suitable ML models.
        Explain briefly why each model is suitable.
        Assume this is a regression problem unless stated otherwise.
        """

        response_text = run_llm(prompt=prompt)

        # 2️ Structured fields (match DB schema)
        problem_type = "regression"

        recommended_models = {
            "recommendations": response_text
        }

        rationale = response_text

        # 3️ Insert into DB (MATCHES TABLE COLUMNS)
        with self.engine.begin() as conn:
            conn.execute(
                sa.text("""
                    INSERT INTO model_recommendations
                    (project_id, problem_type, recommended_models, rationale)
                    VALUES (:pid, :ptype, :models, :rationale)
                """),
                {
                    "pid": self.project_id,
                    "ptype": problem_type,
                    "models": json.dumps(recommended_models),
                    "rationale": rationale,
                }
            )

        return "automl/recommendations_v1"
