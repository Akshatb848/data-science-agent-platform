import sqlalchemy as sa
from utils.db_manager import get_engine
from sklearn.linear_model import LinearRegression
import pandas as pd
import json


class ModelTrainingAgent:
    """
    Trains a simple ML model using engineered features
    and stores model metadata + metrics in the database.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id

    def run(self):
        engine = get_engine()

        # 1️⃣ Load engineered features from DB
        features_df = pd.read_sql(
            "SELECT features FROM feature_store WHERE project_id = %s",
            engine,
            params=(self.project_id,)
        )

        if features_df.empty:
            raise ValueError("No features found in feature_store")

        # 2️⃣ Extract feature JSON (dict)
        feature_dict = features_df.iloc[0]["features"]

        if not isinstance(feature_dict, dict):
            raise TypeError("Features must be a dictionary")

        # 3️⃣ Convert JSON → numeric DataFrame
        X = pd.json_normalize(feature_dict)

        if X.empty:
            raise ValueError("Feature matrix is empty after normalization")

        # 4️⃣ Dummy target (placeholder for now)
        # In real systems, this will come from labels table
        y = [1]

        # 5️⃣ Train model
        model = LinearRegression()
        model.fit(X, y)

        # 6️⃣ Prepare metrics
        metrics = {
            "model": "LinearRegression",
            "num_features": X.shape[1],
            "feature_names": X.columns.tolist(),
            "coefficients": model.coef_.tolist()
        }

        # 7️⃣ Store results in DB
        with engine.begin() as conn:
            conn.execute(
                sa.text("""
                    INSERT INTO model_results
                    (project_id, model_name, metrics)
                    VALUES (:pid, :model, :metrics)
                """),
                {
                    "pid": self.project_id,
                    "model": "LinearRegression",
                    "metrics": json.dumps(metrics)
                }
            )

        # 8️⃣ Return artifact location
        return "models/linear_regression_v1"
