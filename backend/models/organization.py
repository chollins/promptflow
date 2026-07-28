from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin


class Organization(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "organizations"

    name = db.Column(db.String(255), nullable=False)

    slug = db.Column(
        db.String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    code = db.Column(
        db.String(20),
        unique=True,
        nullable=False,
        index=True,
    )

    is_active = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
    )

    users = db.relationship(
        "User",
        back_populates="organization",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    invitations = db.relationship(
        "Invitation",
        back_populates="organization",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    flows = db.relationship(
        "OrganizationFlowAccess",
        back_populates="organization",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
