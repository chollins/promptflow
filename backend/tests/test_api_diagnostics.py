"""
Integration tests for execution & auth endpoints testing diagnostic visibility.
"""
from __future__ import annotations

import pytest
from app import create_app
from config import Config
from models import User, Role
from extensions import db

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"

@pytest.fixture
def app_instance():
    app = create_app(TestConfig)
    app.config["DIAGNOSTICS_SUPERADMIN"] = {"input_sources", "prompts", "model", "output_schema", "raw_response", "structured_output", "execution"}
    app.config["DIAGNOSTICS_ADMIN"] = {"prompts", "execution"}
    app.config["DIAGNOSTICS_USER"] = set()

    with app.app_context():
        db.create_all()

        super_role = Role.query.filter_by(name="superadmin").first() or Role(name="superadmin")
        admin_role = Role.query.filter_by(name="admin").first() or Role(name="admin")
        user_role = Role.query.filter_by(name="user").first() or Role(name="user")

        for r in (super_role, admin_role, user_role):
            if r.id is None:
                db.session.add(r)
        db.session.commit()

        super_user = User(name="Super", email="super_test@example.com", password_hash="dummy_hash", role_id=super_role.id, session_token="token_super_test")
        admin_user = User(name="Admin", email="admin_test@example.com", password_hash="dummy_hash", role_id=admin_role.id, session_token="token_admin_test")
        normal_user = User(name="User", email="user_test@example.com", password_hash="dummy_hash", role_id=user_role.id, session_token="token_user_test")

        db.session.add_all([super_user, admin_user, normal_user])
        db.session.commit()

        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app_instance):
    return app_instance.test_client()

class TestAuthDiagnosticCapabilities:
    def test_auth_me_superadmin_capabilities(self, client):
        res = client.get("/api/auth/me", headers={"X-Session-Token": "token_super_test"})
        assert res.status_code == 200
        data = res.get_json()
        assert "diagnostic_capabilities" in data
        assert set(data["diagnostic_capabilities"]) == {"execution", "input_sources", "model", "output_schema", "prompts", "raw_response", "structured_output"}

    def test_auth_me_admin_capabilities(self, client):
        res = client.get("/api/auth/me", headers={"X-Session-Token": "token_admin_test"})
        assert res.status_code == 200
        data = res.get_json()
        assert "diagnostic_capabilities" in data
        assert set(data["diagnostic_capabilities"]) == {"execution", "prompts"}

    def test_auth_me_user_capabilities(self, client):
        res = client.get("/api/auth/me", headers={"X-Session-Token": "token_user_test"})
        assert res.status_code == 200
        data = res.get_json()
        assert "diagnostic_capabilities" in data
        assert data["diagnostic_capabilities"] == []

    def test_unauthenticated_me(self, client):
        res = client.get("/api/auth/me")
        assert res.status_code == 401

class TestExecutionResponseFormatting:
    def test_error_handler_does_not_leak_stacktrace(self, client):
        res = client.post("/api/forms/nonexistent-id/execute", headers={"X-Session-Token": "token_super_test"})
        assert res.status_code in (404, 500)
        data = res.get_json()
        assert "error" in data
        assert "traceback" not in data
        assert "prompt" not in data
        assert "debug" not in data
