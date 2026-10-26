from __future__ import annotations

import json
import pytest
from app import create_app
from config import Config
from extensions import db
from models import Organization, OrganizationModelLabel, Role, User
from services.model_registry import seed_model_registry


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


@pytest.fixture
def app_instance():
    app = create_app(TestConfig)

    with app.app_context():
        db.create_all()
        seed_model_registry()

        super_role = Role(name="superadmin")
        admin_role = Role(name="admin")
        member_role = Role(name="member")

        db.session.add_all([super_role, admin_role, member_role])
        db.session.commit()

        org1 = Organization(name="Org One", slug="org-one", code="ORG1")
        org2 = Organization(name="Org Two", slug="org-two", code="ORG2")
        db.session.add_all([org1, org2])
        db.session.commit()

        # Users
        superadmin = User(
            name="Super User",
            email="super@example.com",
            password_hash="pw",
            role_id=super_role.id,
            session_token="token_super",
        )
        org1_admin = User(
            name="Org1 Admin",
            email="admin1@example.com",
            password_hash="pw",
            role_id=admin_role.id,
            organization_id=org1.id,
            session_token="token_admin1",
        )
        org1_member = User(
            name="Org1 Member",
            email="member1@example.com",
            password_hash="pw",
            role_id=member_role.id,
            organization_id=org1.id,
            session_token="token_member1",
        )
        org2_admin = User(
            name="Org2 Admin",
            email="admin2@example.com",
            password_hash="pw",
            role_id=admin_role.id,
            organization_id=org2.id,
            session_token="token_admin2",
        )
        org2_member = User(
            name="Org2 Member",
            email="member2@example.com",
            password_hash="pw",
            role_id=member_role.id,
            organization_id=org2.id,
            session_token="token_member2",
        )

        db.session.add_all([superadmin, org1_admin, org1_member, org2_admin, org2_member])
        db.session.commit()

        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app_instance):
    return app_instance.test_client()


def test_org_admin_can_set_and_update_label(client):
    # Org1 Admin sets a label for gpt-4o-mini
    resp = client.post(
        "/api/organization/model-labels",
        headers={"X-Session-Token": "token_admin1"},
        json={"model_id": "gpt-4o-mini", "label": "Org 1 Fast Draft"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["label"] == "Org 1 Fast Draft"
    assert data["model_id"] == "gpt-4o-mini"

    # Update label
    resp_update = client.put(
        "/api/organization/model-labels/gpt-4o-mini",
        headers={"X-Session-Token": "token_admin1"},
        json={"label": "Org 1 Speedy Writer"},
    )
    assert resp_update.status_code == 200
    assert resp_update.get_json()["label"] == "Org 1 Speedy Writer"


def test_members_cannot_modify_labels(client):
    # Org1 Member attempts to set a label -> 403 Forbidden
    resp = client.post(
        "/api/organization/model-labels",
        headers={"X-Session-Token": "token_member1"},
        json={"model_id": "gpt-4o-mini", "label": "Hacked Label"},
    )
    assert resp.status_code == 403
    assert "exclusive to organization admins" in resp.get_json()["error"]

    # Org1 Member attempts to delete a label -> 403 Forbidden
    resp_del = client.delete(
        "/api/organization/model-labels/gpt-4o-mini",
        headers={"X-Session-Token": "token_member1"},
    )
    assert resp_del.status_code == 403


def test_superadmin_cannot_modify_labels(client):
    # Superadmin does not belong to an organization and is not an org admin -> 403 Forbidden
    resp = client.post(
        "/api/organization/model-labels",
        headers={"X-Session-Token": "token_super"},
        json={"model_id": "gpt-4o-mini", "label": "Super Label"},
    )
    assert resp.status_code == 403
    assert "exclusive to organization admins" in resp.get_json()["error"]


def test_unauthenticated_cannot_modify_labels(client):
    resp = client.post(
        "/api/organization/model-labels",
        json={"model_id": "gpt-4o-mini", "label": "Anon Label"},
    )
    assert resp.status_code == 401


def test_members_can_view_labels(client):
    # Admin sets label
    client.post(
        "/api/organization/model-labels",
        headers={"X-Session-Token": "token_admin1"},
        json={"model_id": "gpt-4o-mini", "label": "Internal Fast Assistant"},
    )

    # Member fetches organization model labels
    resp = client.get(
        "/api/organization/model-labels",
        headers={"X-Session-Token": "token_member1"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["labels"]["gpt-4o-mini"] == "Internal Fast Assistant"
    assert len(data["items"]) == 1
    assert data["items"][0]["model_id"] == "gpt-4o-mini"
    assert data["items"][0]["label"] == "Internal Fast Assistant"

    # Member fetches /api/models -> custom_label is populated
    resp_models = client.get(
        "/api/models",
        headers={"X-Session-Token": "token_member1"},
    )
    assert resp_models.status_code == 200
    models_data = resp_models.get_json()
    target_model = next((m for m in models_data["models"] if m["id"] == "gpt-4o-mini"), None)
    assert target_model is not None
    assert target_model["custom_label"] == "Internal Fast Assistant"


def test_isolation_between_organizations(client):
    # Org 1 sets label
    client.post(
        "/api/organization/model-labels",
        headers={"X-Session-Token": "token_admin1"},
        json={"model_id": "gpt-4o-mini", "label": "Org 1 Mini"},
    )

    # Org 2 sets label for same model
    client.post(
        "/api/organization/model-labels",
        headers={"X-Session-Token": "token_admin2"},
        json={"model_id": "gpt-4o-mini", "label": "Org 2 Custom Mini"},
    )

    # Org 1 sees Org 1 label
    resp1 = client.get("/api/organization/model-labels", headers={"X-Session-Token": "token_member1"})
    assert resp1.get_json()["labels"]["gpt-4o-mini"] == "Org 1 Mini"

    # Org 2 sees Org 2 label
    resp2 = client.get("/api/organization/model-labels", headers={"X-Session-Token": "token_member2"})
    assert resp2.get_json()["labels"]["gpt-4o-mini"] == "Org 2 Custom Mini"


def test_delete_label(client):
    client.post(
        "/api/organization/model-labels",
        headers={"X-Session-Token": "token_admin1"},
        json={"model_id": "gpt-4o-mini", "label": "Temporary Label"},
    )

    # Delete
    del_resp = client.delete(
        "/api/organization/model-labels/gpt-4o-mini",
        headers={"X-Session-Token": "token_admin1"},
    )
    assert del_resp.status_code == 200
    assert del_resp.get_json()["deleted"] is True

    # Verify gone
    resp = client.get("/api/organization/model-labels", headers={"X-Session-Token": "token_member1"})
    assert "gpt-4o-mini" not in resp.get_json()["labels"]
