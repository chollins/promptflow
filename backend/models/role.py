from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin


class Role(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "roles"

    name = db.Column(
        db.String(100),
        nullable=False,
        unique=True,
        index=True,
    )

    description = db.Column(
        db.String(255),
        nullable=True,
    )

    is_system = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
    )

    users = db.relationship(
        "User",
        back_populates="role",
        cascade="save-update, merge",
    )

    invitations = db.relationship(
        "Invitation",
        back_populates="role",
        cascade="save-update, merge",
    )

    def __repr__(self):
        return f"<Role {self.name}>"
