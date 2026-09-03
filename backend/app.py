from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

from flask import Flask, request

from config import Config
from extensions import cors, db, migrate
from models import Flow, Invitation, Organization, OrganizationFlowAccess, Role, User, PasswordResetOTP, SavedResult
from routes import api


def _get_cors_origins() -> list[Any]:
    raw = os.getenv("FRONTEND_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    return [
        r"^http://localhost:\d+$",
        r"^http://127\.0\.0\.1:\d+$",
        r"^http://192\.168\.1\.13:\d+$",
    ]


def _origin_is_allowed(origin: str | None) -> bool:
    if not origin:
        return False
    for pattern in _get_cors_origins():
        if isinstance(pattern, str) and pattern.startswith("^"):
            if re.match(pattern, origin):
                return True
        elif origin == pattern:
            return True
    return False

def create_app(config_object: type[Config] | None = None) -> Flask:
    app = Flask(__name__)
    
    app.config.from_object(config_object or Config)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret-key")
    
    from services.diagnostics import parse_diagnostic_config
    app.config["DIAGNOSTICS_SUPERADMIN"] = parse_diagnostic_config(os.getenv("DIAGNOSTICS_SUPERADMIN"), "all")
    app.config["DIAGNOSTICS_ADMIN"] = parse_diagnostic_config(os.getenv("DIAGNOSTICS_ADMIN"), "none")
    app.config["DIAGNOSTICS_USER"] = parse_diagnostic_config(os.getenv("DIAGNOSTICS_USER"), "none")

    # Log enabled categories per role at startup (values are never logged, only names)
    for role_key in ("DIAGNOSTICS_SUPERADMIN", "DIAGNOSTICS_ADMIN", "DIAGNOSTICS_USER"):
        cats = app.config[role_key]
        role_label = role_key.replace("DIAGNOSTICS_", "").lower()
        logger.info(
            "Diagnostic policy startup role=%s categories=%s",
            role_label,
            sorted(cats) if cats else "none",
        )


    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(
        app,
        supports_credentials=True,
        resources={r"/api/*": {"origins": _get_cors_origins()}},
        allow_headers=["Content-Type", "X-Session-Token", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )


    @app.after_request
    def _set_cors_headers(response):
        origin = request.headers.get("Origin")
        if _origin_is_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-Token, Authorization"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            response.headers.add("Vary", "Origin")
            
        return response

    # Ensure Alembic sees all models through the app import path.
    _ = (Flow, Invitation, Organization, OrganizationFlowAccess, Role, User, PasswordResetOTP)

    @app.route("/api/version", methods=["GET"])
    def version():
        return {
            "git_sha": os.getenv("RENDER_GIT_COMMIT"),
            "version": os.getenv("RENDER_GIT_COMMIT") or "dev",
        }
    from werkzeug.exceptions import HTTPException
    from flask import jsonify

    @app.errorhandler(Exception)
    def _handle_exception(e):
        if isinstance(e, HTTPException):
            return jsonify({"error": e.description}), e.code
        logger.exception("Unhandled error during request processing")
        return jsonify({"error": "An internal server error occurred."}), 500

    app.register_blueprint(api)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
