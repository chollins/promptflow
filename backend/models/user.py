from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin


class User(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    organization_id = db.Column(
        db.String(36),
        db.ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
    )

    email = db.Column(
        db.String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False,
    )

    name = db.Column(
        db.String(255),
        nullable=False,
    )
    role_id = db.Column(
        db.String(36),
        db.ForeignKey("roles.id"),
        nullable=False,
    )

    role = db.relationship(
        "Role",
        back_populates="users",
    )
    is_active = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
    )

    session_token = db.Column(
        db.String(64),
        unique=True,
        index=True,
        nullable=True,
    )

    organization = db.relationship(
        "Organization",
        back_populates="users",
    )

    invitations_created = db.relationship(
        "Invitation",
        foreign_keys="Invitation.created_by",
        back_populates="creator",
        cascade="all, delete-orphan",
    )   
