"""
Tests for Saved Results feature (models, service, permissions, and API endpoints).
"""
from __future__ import annotations

import json
import pytest
from app import create_app
from config import Config
from models import User, Role, Organization, SavedResult
from extensions import db

class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"

@pytest.fixture
def app_instance():
    app = create_app(TestConfig)
    app.config["SAVED_RESULTS_AUTO_SAVE"] = True
    app.config["SAVED_RESULTS_ORG_ADMIN_ACCESS"] = True

    with app.app_context():
        db.create_all()

        super_role = Role.query.filter_by(name="superadmin").first() or Role(name="superadmin")
        admin_role = Role.query.filter_by(name="admin").first() or Role(name="admin")
        user_role = Role.query.filter_by(name="user").first() or Role(name="user")

        for r in (super_role, admin_role, user_role):
            if r.id is None:
                db.session.add(r)
        db.session.commit()

        org1 = Organization(name="Org 1", slug="org-1", code="ORG1")
        org2 = Organization(name="Org 2", slug="org-2", code="ORG2")
        db.session.add_all([org1, org2])
        db.session.commit()

        super_user = User(name="Super", email="super_sr@example.com", password_hash="pw", role_id=super_role.id, session_token="tok_super")
        admin1_user = User(name="Admin 1", email="admin1_sr@example.com", password_hash="pw", role_id=admin_role.id, organization_id=org1.id, session_token="tok_admin1")
        user1_user = User(name="User 1", email="user1_sr@example.com", password_hash="pw", role_id=user_role.id, organization_id=org1.id, session_token="tok_user1")
        user2_user = User(name="User 2", email="user2_sr@example.com", password_hash="pw", role_id=user_role.id, organization_id=org2.id, session_token="tok_user2")

        db.session.add_all([super_user, admin1_user, user1_user, user2_user])
        db.session.commit()

        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app_instance):
    return app_instance.test_client()

class TestSavedResultsPermissions:
    def test_saved_results_hierarchy(self, client, app_instance):
        with app_instance.app_context():
            u1 = User.query.filter_by(email="user1_sr@example.com").first()
            u2 = User.query.filter_by(email="user2_sr@example.com").first()
            from services.saved_result_service import save_execution_result

            res1 = save_execution_result(
                user_id=u1.id,
                organization_id=u1.organization_id,
                source_type="form",
                source_id="f1",
                source_name="Form 1",
                input_summary={"input": "val1"},
                output_text="Result 1 for User 1",
            )
            res2 = save_execution_result(
                user_id=u2.id,
                organization_id=u2.organization_id,
                source_type="flow",
                source_id="flow1",
                source_name="Flow 1",
                input_summary={"input": "val2"},
                output_text="Result 2 for User 2",
            )
            res1_id = res1.id

        # 1. User 1 can only see own result (res1)
        res = client.get("/api/saved-results", headers={"X-Session-Token": "tok_user1"})
        assert res.status_code == 200
        items = res.get_json()["items"]
        assert len(items) == 1
        assert items[0]["id"] == res1_id

        # 2. Admin 1 can see all results in Org 1 (res1)
        res = client.get("/api/saved-results", headers={"X-Session-Token": "tok_admin1"})
        assert res.status_code == 200
        items = res.get_json()["items"]
        assert len(items) == 1
        assert items[0]["id"] == res1_id

        # 3. Superadmin sees all results (res1 & res2)
        res = client.get("/api/saved-results", headers={"X-Session-Token": "tok_super"})
        assert res.status_code == 200
        items = res.get_json()["items"]
        assert len(items) == 2

    def test_user_cannot_delete_other_user_result(self, client, app_instance):
        with app_instance.app_context():
            u2 = User.query.filter_by(email="user2_sr@example.com").first()
            from services.saved_result_service import save_execution_result

            res2 = save_execution_result(
                user_id=u2.id,
                organization_id=u2.organization_id,
                source_type="form",
                source_id="f2",
                source_name="Form 2",
                input_summary={},
                output_text="User 2 text",
            )
            res2_id = res2.id

        # User 1 attempts delete
        res = client.delete(f"/api/saved-results/{res2_id}", headers={"X-Session-Token": "tok_user1"})
        assert res.status_code == 404

        # Superadmin deletes
        res = client.delete(f"/api/saved-results/{res2_id}", headers={"X-Session-Token": "tok_super"})
        assert res.status_code == 200
        assert res.get_json() == {"ok": True}

    def test_deduplication_of_saved_results(self, client, app_instance):
        with app_instance.app_context():
            u1 = User.query.filter_by(email="user1_sr@example.com").first()
            from services.saved_result_service import save_execution_result

            r1 = save_execution_result(
                user_id=u1.id,
                organization_id=u1.organization_id,
                source_type="form",
                source_id="f_dedup",
                source_name="Dedup Form",
                input_summary={"q": "test"},
                output_text="Same output text",
            )
            r1_id = r1.id

            # Save exact same again
            r2 = save_execution_result(
                user_id=u1.id,
                organization_id=u1.organization_id,
                source_type="form",
                source_id="f_dedup",
                source_name="Dedup Form",
                input_summary={"q": "test"},
                output_text="Same output text",
            )
            # Should return the same record ID without creating a duplicate
            assert r2.id == r1_id

            # Total count for u1 should be 1
            all_res = SavedResult.query.filter_by(user_id=u1.id, source_id="f_dedup").all()
            assert len(all_res) == 1
