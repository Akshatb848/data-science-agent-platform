"""
Platform configuration — environment-driven settings for all modules.
"""

import os
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Central configuration for TennisIQ platform."""

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "TennisIQ"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True

    # ── API ──────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["*"]

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./tennisiq.db"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Auth ─────────────────────────────────────────────
    JWT_SECRET: str = Field(default="tennisiq-dev-secret-change-me")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24

    # ── Storage ──────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    MAX_VIDEO_SIZE_MB: int = 2048
    SUPPORTED_VIDEO_FORMATS: list[str] = [".mp4", ".mov", ".m4v"]

    # ── ML / AI ──────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = "gpt-4o"
    ML_MODELS_DIR: str = "./ml_models"
    INFERENCE_DEVICE: str = "cpu"  # cpu | coreml | cuda

    # ── Video Processing ─────────────────────────────────
    TARGET_FPS: int = 30
    FRAME_WIDTH: int = 1920
    FRAME_HEIGHT: int = 1080

    # ── Tennis Defaults ──────────────────────────────────
    DEFAULT_MATCH_FORMAT: str = "best_of_3"
    DEFAULT_TIEBREAK_AT: int = 6
    DEFAULT_FINAL_SET_TIEBREAK: bool = True

    # ── Subscription ─────────────────────────────────────
    FREE_SESSIONS_PER_MONTH: int = 3
    FREE_VIDEO_WATERMARK: bool = True
    PRO_PRICE_MONTHLY: float = 9.99
    ELITE_PRICE_MONTHLY: float = 24.99

    # ── Streaming ────────────────────────────────────────
    HLS_SEGMENT_DURATION: int = 6
    CDN_BASE_URL: str = ""
    RTMP_INGEST_URL: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
