from __future__ import annotations

import os
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:

    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(BASE_DIR / 'app.db').as_posix()}",
    )
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_SORT_KEYS = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_PATH = "/"
    SESSION_COOKIE_NAME = "promptflow_session"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1", "yes")
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)

    SAVED_RESULTS_AUTO_SAVE = os.getenv("SAVED_RESULTS_AUTO_SAVE", "true").lower() in ("true", "1", "yes")
    SAVED_RESULTS_ORG_ADMIN_ACCESS = os.getenv("SAVED_RESULTS_ORG_ADMIN_ACCESS", "true").lower() in ("true", "1", "yes")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "true").lower() in ("true", "1", "yes")
