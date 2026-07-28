from __future__ import annotations

from extensions import db
from .base import TimestampMixin


class OrganizationFlowAccess(db.Model, TimestampMixin):
    __tablename__ = "organization_flow_access"

    organization_id = db.Column(
        db.String(36),
        db.ForeignKey("organizations.id", ondelete="CASCADE"),
        primary_key=True,
    )

    flow_id = db.Column(
        db.String(36),
        db.ForeignKey("flows.id", ondelete="CASCADE"),
        primary_key=True,
    )

    organization = db.relationship(
        "Organization",
        back_populates="flows",
    )

    flow = db.relationship(
        "Flow",
        back_populates="organizations",
    )
