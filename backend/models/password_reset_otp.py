from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin

class PasswordResetOTP(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "password_reset_otps"

    user_id = db.Column(
        db.String(36),
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    otp_hash = db.Column(
        db.String(255),
        nullable=False,
    )
    expires_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
    )
    attempts = db.Column(
        db.Integer,
        nullable=False,
        default=0,
    )
    verified_at = db.Column(
        db.DateTime(timezone=True),
        nullable=True,
    )

    user = db.relationship(
        "User",
        backref=db.backref("otps", cascade="all, delete-orphan"),
    )
