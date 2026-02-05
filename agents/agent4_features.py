import pandas as pd
import sqlalchemy as sa
import json

from sqlalchemy.dialects.postgresql import JSONB
from utils.db_manager import get_engine


class FeatureEngineeringAgent:
    """
    Agent 4: Feature Engineering Agent
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.engine = get_engine()

    def _init_table(self):
        """
        Create feature_store table if not exists
        """
        with self.engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS feature_store (
                    id SERIAL PRIMARY KEY,
                    project_id VARCHAR(50),
                    features JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))

    def _load_cleaned_data(self) -> pd.DataFrame:
        """
        Load latest cleaned data for the project
        """
        query = """
            SELECT data
            FROM cleaned_data
            WHERE project_id = :project_id
            ORDER BY created_at DESC
            LIMIT 1
        """

        df = pd.read_sql(
            sa.text(query),
            self.engine,
            params={"project_id": self.project_id},
        )

        if df.empty:
            raise ValueError("No cleaned data found for project")

        # JSONB → DataFrame
        records = df.iloc[0]["data"]
        return pd.DataFrame(records)

    def _engineer_features(self, df: pd.DataFrame) -> dict:
        """
        Simple deterministic feature engineering
        """
        features = {}

        # Numeric features
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            features[f"{col}_mean"] = float(df[col].mean())
            features[f"{col}_min"] = float(df[col].min())
            features[f"{col}_max"] = float(df[col].max())

        # Categorical features
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts().to_dict()
            features[f"{col}_counts"] = value_counts

        return features

    def run(self) -> str:
        """
        Execute feature engineering pipeline
        """
        self._init_table()

        df = self._load_cleaned_data()
        features = self._engineer_features(df)

        stmt = sa.text("""
            INSERT INTO feature_store (project_id, features)
            VALUES (:project_id, :features)
        """).bindparams(
            sa.bindparam("features", type_=JSONB)
        )

        with self.engine.begin() as conn:
            conn.execute(
                stmt,
                {
                    "project_id": self.project_id,
                    "features": features,
                }
            )

        return "feature_store/v1"
