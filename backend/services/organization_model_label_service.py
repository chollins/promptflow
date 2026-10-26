from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from extensions import db
from models import OrganizationModelLabel
from services.model_registry import is_valid_model

logger = logging.getLogger(__name__)


def get_organization_model_labels(organization_id: Optional[str]) -> Dict[str, str]:
    """Returns a dictionary mapping model_id to custom label for the given organization."""
    if not organization_id:
        return {}
    try:
        records = OrganizationModelLabel.query.filter_by(organization_id=organization_id).all()
        return {item.model_id: item.label for item in records}
    except Exception as exc:
        logger.warning("Failed to fetch organization model labels for org %s: %s", organization_id, exc)
        return {}


def get_organization_model_label_items(organization_id: Optional[str]) -> List[Dict[str, Any]]:
    """Returns a list of dict representation of custom labels for the given organization."""
    if not organization_id:
        return []
    try:
        records = OrganizationModelLabel.query.filter_by(organization_id=organization_id).order_by(OrganizationModelLabel.created_at.asc()).all()
        return [item.to_dict() for item in records]
    except Exception as exc:
        logger.warning("Failed to fetch organization model label items for org %s: %s", organization_id, exc)
        return []


def set_organization_model_label(organization_id: str, model_id: str, label: str) -> OrganizationModelLabel:
    """Creates or updates a custom label for an AI model in an organization."""
    if not organization_id or not organization_id.strip():
        raise ValueError("Organization ID is required.")

    if not model_id or not model_id.strip():
        raise ValueError("Model ID is required.")

    clean_model_id = model_id.strip()
    clean_label = (label or "").strip()

    if not clean_label:
        raise ValueError("Label cannot be empty.")

    if len(clean_label) > 255:
        raise ValueError("Label cannot exceed 255 characters.")

    if not is_valid_model(clean_model_id):
        raise ValueError(f"Unknown or invalid AI model '{clean_model_id}'.")

    existing = OrganizationModelLabel.query.filter_by(
        organization_id=organization_id,
        model_id=clean_model_id,
    ).first()

    if existing:
        existing.label = clean_label
        record = existing
    else:
        record = OrganizationModelLabel(
            organization_id=organization_id,
            model_id=clean_model_id,
            label=clean_label,
        )
        db.session.add(record)

    db.session.commit()
    return record


def delete_organization_model_label(organization_id: str, model_id: str) -> bool:
    """Deletes a custom label for an AI model in an organization. Returns True if deleted, False if not found."""
    if not organization_id or not model_id:
        return False

    record = OrganizationModelLabel.query.filter_by(
        organization_id=organization_id,
        model_id=model_id.strip(),
    ).first()

    if not record:
        return False

    db.session.delete(record)
    db.session.commit()
    return True
