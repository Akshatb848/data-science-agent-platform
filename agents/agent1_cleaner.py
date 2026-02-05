import json 
import pandas as pd
import sqlalchemy as sa

from sqlalchemy.dialects.postgresql import JSONB
from utils.db_manager import get_engine


class DataCleanerAgent:
    """
    Agent 1: Data Cleaner & Processing Agent
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.engine = get_engine()

    def _init_table(self):
        """
        Create cleaned_data table if not exists.
        """
        with self.engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS cleaned_data (
                    id SERIAL PRIMARY KEY,
                    project_id VARCHAR(50),
                    data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))

    def clean(self, df: pd.DataFrame) -> str:
        """
        Clean the input DataFrame and store result in DB.
        Returns output_location.
        """
        self._init_table()

        # ---- Cleaning steps ----
        df = df.copy()

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Handle missing values (simple strategy for v1)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("unknown")
            else:
                df[col] = df[col].fillna(0)

        # Convert to JSON records
        records = df.to_dict(orient="records")

        json_data = json.dumps(records)
        # ---- Store in DB ----
        
        stmt = sa.text("""
        INSERT INTO cleaned_data (project_id, data)
        VALUES (:project_id, :data)
    """).bindparams(
        sa.bindparam("data", type_=JSONB)
    )
        
        with self.engine.begin() as conn:
            conn.execute(
            stmt,
            {
                "project_id": self.project_id,
                "data": records,  # pass Python object directly
            }
    )

        return "cleaned_data/latest"
