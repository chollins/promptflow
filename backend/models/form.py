from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin

class Form(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "forms"

    name = db.Column(
        db.String(255),
        nullable=False,
    )

    slug = db.Column(
        db.String(150),
        nullable=False,
        unique=True,
    )

    description = db.Column(
        db.Text,
        nullable=True,
    )

    content_json = db.Column(
        db.Text,
        nullable=True,
    )

    file_path = db.Column(
        db.String(500),
        nullable=False,
    )

    is_active = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
    )


    flow_steps = db.relationship(
        "FlowFormStep",
        back_populates="form",
        cascade="all, delete-orphan",
    )