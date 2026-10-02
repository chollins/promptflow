from extensions import db
from .base import TimestampMixin, UUIDMixin


class AIModel(db.Model, UUIDMixin, TimestampMixin):
    __tablename__ = "ai_models"

    model_id = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    provider = db.Column(db.String(100), nullable=False, default="openai")
    category = db.Column(db.String(100), nullable=False, default="text")
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    is_recommended = db.Column(db.Boolean, nullable=False, default=False)
