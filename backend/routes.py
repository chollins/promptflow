from __future__ import annotations

from flask import Blueprint, jsonify

api = Blueprint("api", __name__)


@api.get("/health")
def health_check():
    return jsonify({"status": "ok"})


@api.get("/api")
def api_root():
    return jsonify(
        {
            "message": "PromptFlow backend is running.",
            "endpoints": ["/health", "/api/forms", "/api/flows"],
        }
    )


@api.get("/api/forms")
def list_forms():
    return jsonify({"items": [], "count": 0})


@api.get("/api/flows")
def list_flows():
    return jsonify({"items": [], "count": 0})
