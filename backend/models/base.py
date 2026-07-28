from __future__ import annotations

import uuid

from sqlalchemy import text

from extensions import db


class TimestampMixin:
    created_at = db.Column(
        db.DateTime,
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    updated_at = db.Column(
        db.DateTime,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


class UUIDMixin:
    id = db.Column(
        db.String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
