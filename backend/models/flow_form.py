from __future__ import annotations

from extensions import db
from .base import TimestampMixin, UUIDMixin


class FlowFormStep(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "flow_form_steps"

    flow_id = db.Column(
        db.ForeignKey("flows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    form_id = db.Column(
        db.ForeignKey("forms.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
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

    __table_args__ = (
        db.UniqueConstraint(
            "flow_id",
            "step_number",
            name="uq_flow_step_number",
        ),
    )