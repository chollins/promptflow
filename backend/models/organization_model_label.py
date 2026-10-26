from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin


class OrganizationModelLabel(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "organization_model_labels"

    organization_id = db.Column(
        db.String(36),
        db.ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_id = db.Column(
        db.String(255),
        nullable=False,
        index=True,
    )

    label = db.Column(
        db.String(255),
        nullable=False,
    )

    __table_args__ = (
        db.UniqueConstraint("organization_id", "model_id", name="uq_org_model_label"),
    )

    organization = db.relationship(
        "Organization",
        back_populates="model_labels",
    )

    def to_dict(self):
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "model_id": self.model_id,
            "label": self.label,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
