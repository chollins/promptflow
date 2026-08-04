from __future__ import annotations

import os

from flask import Flask

from config import Config
from extensions import cors, db, migrate
from models import Flow, Invitation, Organization, OrganizationFlowAccess, Role, User
from routes import api


def _get_cors_origins() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.13:8080",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

def create_app(config_object: type[Config] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object or Config)

    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(
        app,
        supports_credentials=True,
        origins=_get_cors_origins(),
    )

    # Ensure Alembic sees all models through the app import path.
    _ = (Flow, Invitation, Organization, OrganizationFlowAccess, Role, User)

    app.register_blueprint(api)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
