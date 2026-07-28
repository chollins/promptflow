from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin


class Invitation(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "invitations"

    organization_id = db.Column(
        db.String(36),
        db.ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
    )

    created_by = db.Column(
        db.String(36),
        db.ForeignKey("users.id"),
        nullable=False,
    )

    email = db.Column(
        db.String(255),
        nullable=False,
        index=True,
    )

    role_id = db.Column(
        db.String(36),
        db.ForeignKey("roles.id"),
        nullable=False,
    )

    role = db.relationship(
        "Role",
        back_populates="invitations",
    )
    token_hash = db.Column(
        db.String(64),
        nullable=False,
        unique=True,
    )

    expires_at = db.Column(
        db.DateTime,
        nullable=False,
    )

    accepted_at = db.Column(
        db.DateTime,
        nullable=True,
    )

    organization = db.relationship(
        "Organization",
        back_populates="invitations",
    )

    creator = db.relationship(
        "User",
        foreign_keys=[created_by],
        back_populates="invitations_created",
    )
