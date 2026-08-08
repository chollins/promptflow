from __future__ import annotations

from extensions import db
from .base import TimestampMixin


class FlowFormStep(db.Model, TimestampMixin):
    __tablename__ = "flow_form_steps"

    flow_id = db.Column(
        db.String(36),
        db.ForeignKey("flows.id", ondelete="CASCADE"),
        primary_key=True,
    )

    form_id = db.Column(
        db.String(36),
        db.ForeignKey("forms.id", ondelete="CASCADE"),
        primary_key=True,
    )

    step_number = db.Column(
        db.Integer,
        nullable=False,
    )

    is_required = db.Column(
        db.Boolean,
        nullable=False,
        default=True,
    )

    flow = db.relationship(
        "Flow",
        back_populates="form_steps",
    )

    form = db.relationship(
        "Form",
        back_populates="flow_steps",
    )
