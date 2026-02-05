import os
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from datetime import datetime

# ✅ DEFINE ENGINE AT MODULE LEVEL
_ENGINE = None


def get_engine():
    global _ENGINE

    if _ENGINE is None:
        db_url = os.getenv("DATABASE_URL")

        if not db_url:
            raise RuntimeError(
                "DATABASE_URL environment variable not set. "
                "Please configure Neon DB in Streamlit secrets."
            )

        _ENGINE = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )

    return _ENGINE


def ensure_project_exists(project_id: str, description: str = ""):
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.execute(text("""
            INSERT INTO projects (project_id, description)
            VALUES (:project_id, :description)
            ON CONFLICT (project_id) DO NOTHING
        """), {
            "project_id": project_id,
            "description": description
        })


def log_agent_start(project_id: str, agent_name: str):
    engine = get_engine()
    ensure_project_exists(project_id)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO agent_logs (
                project_id, agent_name, status, started_at
            )
            VALUES (:project_id, :agent_name, 'running', :started_at)
        """), {
            "project_id": project_id,
            "agent_name": agent_name,
            "started_at": datetime.utcnow()
        })


def log_agent_success(project_id: str, agent_name: str):
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE agent_logs
            SET status = 'completed', completed_at = :completed_at
            WHERE project_id = :project_id
              AND agent_name = :agent_name
              AND status = 'running'
        """), {
            "project_id": project_id,
            "agent_name": agent_name,
            "completed_at": datetime.utcnow()
        })
