from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import pytest

from app import create_app
from config import Config
from extensions import db
from models import Organization, PasswordResetOTP, Role, User
from routes import _ensure_utc, _utcnow


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SECRET_KEY = "test-secret"


@pytest.fixture
def app_instance():
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()

        role = Role.query.filter_by(name="user").first() or Role(name="user")
        if role.id is None:
            db.session.add(role)
        db.session.commit()

        org = Organization(name="Test Org", slug="test-org", code="TORG")
        db.session.add(org)
        db.session.commit()

        user = User(
            name="Test User",
            email="test@example.com",
            password_hash="pw_hash",
            role_id=role.id,
            organization_id=org.id,
            is_active=True,
        )
        db.session.add(user)
        db.session.commit()

        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app_instance):
    return app_instance.test_client()


def test_utcnow_is_timezone_aware():
    now = _utcnow()
    assert now.tzinfo is not None
    assert now.tzinfo.utcoffset(now) == timedelta(0)


def test_ensure_utc():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    aware_utc = _ensure_utc(naive)
    assert aware_utc.tzinfo == timezone.utc

    offset_plus_8 = timezone(timedelta(hours=8))
    aware_plus_8 = datetime(2026, 1, 1, 20, 0, 0, tzinfo=offset_plus_8)
    converted = _ensure_utc(aware_plus_8)
    assert converted.tzinfo == timezone.utc
    assert converted.hour == 12

    assert _ensure_utc(None) is None


def test_verify_otp_with_aware_and_naive_datetimes(client, app_instance):
    otp_code = "123456"
    otp_hash = sha256(otp_code.encode("utf-8")).hexdigest()

    with app_instance.app_context():
        user = User.query.filter_by(email="test@example.com").first()

        # 1. Test with timezone-aware datetime (e.g. from PostgreSQL TIMESTAMPTZ)
        aware_expiry = datetime.now(timezone.utc) + timedelta(minutes=10)
        otp_aware = PasswordResetOTP(
            user_id=user.id,
            otp_hash=otp_hash,
            expires_at=aware_expiry,
            attempts=0,
        )
        db.session.add(otp_aware)
        db.session.commit()

    # Request verify-otp
    resp = client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": "test@example.com", "otp": otp_code},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") is True
    assert "reset_token" in data

    # 2. Test with naive datetime (e.g. legacy records or SQLite)
    with app_instance.app_context():
        user = User.query.filter_by(email="test@example.com").first()
        PasswordResetOTP.query.filter_by(user_id=user.id).delete()
        naive_expiry = datetime.utcnow() + timedelta(minutes=10)
        otp_naive = PasswordResetOTP(
            user_id=user.id,
            otp_hash=otp_hash,
            expires_at=naive_expiry,
            attempts=0,
        )
        db.session.add(otp_naive)
        db.session.commit()

    resp = client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": "test@example.com", "otp": otp_code},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") is True


def test_verify_otp_expired(client, app_instance):
    otp_code = "654321"
    otp_hash = sha256(otp_code.encode("utf-8")).hexdigest()

    with app_instance.app_context():
        user = User.query.filter_by(email="test@example.com").first()
        # Expired aware datetime
        expired_expiry = datetime.now(timezone.utc) - timedelta(minutes=1)
        otp = PasswordResetOTP(
            user_id=user.id,
            otp_hash=otp_hash,
            expires_at=expired_expiry,
            attempts=0,
        )
        db.session.add(otp)
        db.session.commit()

    resp = client.post(
        "/api/auth/forgot-password/verify-otp",
        json={"email": "test@example.com", "otp": otp_code},
    )
    assert resp.status_code == 400
    assert resp.get_json().get("error") == "OTP has expired."
