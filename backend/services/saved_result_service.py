from __future__ import annotations

import json
import logging
from flask import current_app
from extensions import db
from models import SavedResult, User, Organization

logger = logging.getLogger(__name__)

def save_execution_result(
    user_id: str,
    organization_id: str | None,
    source_type: str,  # "form" or "flow"
    source_id: str,
    source_name: str,
    input_summary: dict | None,
    output_text: str,
    output_json: dict | list | None = None,
) -> SavedResult:
    """Save or update an execution result into the database to prevent duplicate entries."""
    from datetime import datetime, timezone

    input_summary_json = json.dumps(input_summary) if input_summary is not None else None
    output_json_str = json.dumps(output_json) if output_json is not None else None

    # Check for existing result with same user, source, and either exact output or exact inputs
    existing = SavedResult.query.filter_by(
        user_id=user_id,
        source_type=source_type,
        source_id=source_id,
        output_text=output_text,
    ).first()

    if not existing and input_summary_json is not None:
        existing = SavedResult.query.filter_by(
            user_id=user_id,
            source_type=source_type,
            source_id=source_id,
            input_summary_json=input_summary_json,
        ).first()

    if existing:
        existing.organization_id = organization_id
        existing.source_name = source_name
        existing.input_summary_json = input_summary_json
        existing.output_text = output_text
        existing.output_json = output_json_str
        existing.created_at = datetime.now(timezone.utc)
        db.session.commit()
        logger.info("Updated existing saved result id=%s source_type=%s source_name=%s user_id=%s", existing.id, source_type, source_name, user_id)
        return existing

    result = SavedResult(
        user_id=user_id,
        organization_id=organization_id,
        source_type=source_type,
        source_id=source_id,
        source_name=source_name,
        input_summary_json=input_summary_json,
        output_text=output_text,
        output_json=output_json_str,
    )
    db.session.add(result)
    db.session.commit()
    logger.info("Saved new execution result id=%s source_type=%s source_name=%s user_id=%s", result.id, source_type, source_name, user_id)
    return result


def get_user_access_filter(current_user: User):
    """
    Returns SQL alchemy filter expression based on user role and configuration:
    - Regular User: only own results
    - Org Admin: all results in org if SAVED_RESULTS_ORG_ADMIN_ACCESS=True, else only own
    - Superadmin: all results
    """
    role_name = current_user.role.name.lower() if current_user.role else "user"
    org_admin_access = current_app.config.get("SAVED_RESULTS_ORG_ADMIN_ACCESS", True)

    if role_name == "superadmin":
        return None  # No filter (all results)

    if role_name in ("admin", "org_admin") and org_admin_access and current_user.organization_id:
        return SavedResult.organization_id == current_user.organization_id

    # Fallback / regular user: strictly own results
    return SavedResult.user_id == current_user.id


def list_saved_results(
    current_user: User,
    source_type: str | None = None,
    search: str | None = None,
) -> list[dict]:
    query = SavedResult.query

    access_filter = get_user_access_filter(current_user)
    if access_filter is not None:
        query = query.filter(access_filter)

    if source_type in ("form", "flow"):
        query = query.filter(SavedResult.source_type == source_type)

    if search:
        search_pattern = f"%{search.strip()}%"
        query = query.filter(
            (SavedResult.source_name.ilike(search_pattern)) |
            (SavedResult.output_text.ilike(search_pattern))
        )

    results = query.order_by(SavedResult.created_at.desc()).all()
    return [r.to_dict() for r in results]


def get_saved_result_by_id(result_id: str, current_user: User) -> SavedResult | None:
    result = db.session.get(SavedResult, result_id)
    if not result:
        return None

    role_name = current_user.role.name.lower() if current_user.role else "user"
    org_admin_access = current_app.config.get("SAVED_RESULTS_ORG_ADMIN_ACCESS", True)

    if role_name == "superadmin":
        return result

    if role_name in ("admin", "org_admin") and org_admin_access and current_user.organization_id:
        if result.organization_id == current_user.organization_id:
            return result

    if result.user_id == current_user.id:
        return result

    return None


def delete_saved_result(result_id: str, current_user: User) -> bool:
    result = get_saved_result_by_id(result_id, current_user)
    if not result:
        return False

    db.session.delete(result)
    db.session.commit()
    logger.info("Deleted saved result id=%s by user_id=%s", result_id, current_user.id)
    return True
