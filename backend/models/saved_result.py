from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin

class SavedResult(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "saved_results"

    user_id = db.Column(
        db.String(36),
        db.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    organization_id = db.Column(
        db.String(36),
        db.ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
    )

    source_type = db.Column(
        db.String(50),  # "form" or "flow"
        nullable=False,
    )

    source_id = db.Column(
        db.String(255),
        nullable=False,
    )

    source_name = db.Column(
        db.String(255),
        nullable=False,
    )

    input_summary_json = db.Column(
        db.Text,
        nullable=True,
    )

    output_text = db.Column(
        db.Text,
        nullable=False,
    )

    output_json = db.Column(
        db.Text,
        nullable=True,
    )

    user = db.relationship("User", backref=db.backref("saved_results", cascade="all, delete-orphan"))
    organization = db.relationship("Organization", backref=db.backref("saved_results", cascade="all, delete-orphan"))

    def to_dict(self) -> dict:
        import json
        inputs = None
        if self.input_summary_json:
            try:
                inputs = json.loads(self.input_summary_json)
            except Exception:
                inputs = self.input_summary_json

        out_json = None
        if self.output_json:
            try:
                out_json = json.loads(self.output_json)
            except Exception:
                out_json = self.output_json

        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user.name if self.user else "Unknown User",
            "organization_id": self.organization_id,
            "organization_name": self.organization.name if self.organization else None,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "input_summary": inputs,
            "output_text": self.output_text,
            "output_json": out_json,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }
