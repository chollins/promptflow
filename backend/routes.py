from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
from uuid import uuid4

from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db
from models import Flow, Invitation, Organization, OrganizationFlowAccess, Role, User
from services.flow_executor import execute_flow
from services.flow_service import (
    FlowNotFoundError,
    InvalidFlowError,
    create_flow,
    delete_flow,
    get_all_flows as list_all_flows,
    get_flow as load_flow,
    update_flow,
)
from services.form_executor import execute_form
from services.form_service import (
    FormNotFoundError,
    InvalidFormError,
    get_all_forms as list_all_forms,
    get_form as load_form,
)

api = Blueprint("api", __name__)


def _get_current_user() -> User | None:
    token = request.headers.get("X-Session-Token")
    if token:
        user = User.query.filter_by(session_token=token, is_active=True).first()
        if user:
            return user

    user_id = session.get("user_id")
    if not user_id:
        return None

    return User.query.get(user_id)


def _get_frontend_base_url() -> str:
    origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
    if origin.startswith("http://") or origin.startswith("https://"):
        return origin.rstrip("/")
    return "http://localhost:8080"


def _build_invitation_link(token: str) -> str:
    return f"{_get_frontend_base_url()}/signup?token={token}"


def _utcnow() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@api.post("/api/auth/login")
def auth_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email, is_active=True).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = session.get("session_token") or str(uuid4())
    session["user_id"] = user.id
    session["session_token"] = token
    user.session_token = token
    db.session.commit()
    return jsonify({
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role.name if user.role else None,
        "organization_id": user.organization_id,
        "session_token": session["session_token"],
    })


@api.post("/api/auth/logout")
def auth_logout():
    current_user = _get_current_user()
    if current_user:
        current_user.session_token = None
        db.session.commit()
    session.clear()
    return jsonify({"ok": True})


@api.get("/api/auth/me")
def auth_me():
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401

    if not user or not user.is_active:
        session.clear()
        return jsonify({"error": "Not authenticated"}), 401

    return jsonify({
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role.name if user.role else None,
        "organization_id": user.organization_id,
        "session_token": session.get("session_token"),
    })


# ---------------------------------------------------------------------------

@api.get("/health")
def health_check():
    return jsonify({"status": "ok"})


@api.get("/api")
def api_root():
    return jsonify(
        {
            "message": "PromptFlow backend is running.",
            "endpoints": ["/health", "/api/forms", "/api/flows"],
        }
    )


@api.get("/api/forms")
def list_forms():
    items = [form.model_dump() for form in list_all_forms()]
    return jsonify({"items": items, "count": len(items)})


@api.get("/api/forms/<form_id>")
def get_form(form_id: str):
    form = load_form(form_id)
    return jsonify(form.model_dump())


@api.post("/api/forms/<form_id>/execute")
def execute_form_route(form_id: str):
    payload = request.get_json(silent=True) or {}
    result = execute_form(
        form_id=form_id,
        values=payload.get("values") or {},
    )
    return jsonify(result.model_dump())


@api.get("/api/flows")
def list_flows():
    items = [flow.model_dump() for flow in list_all_flows()]
    return jsonify({"items": items, "count": len(items)})


@api.get("/api/flows/<flow_id>")
def get_flow(flow_id: str):
    flow = load_flow(flow_id)
    return jsonify(flow.model_dump())


@api.post("/api/flows/<flow_id>/execute")
def run_flow(flow_id: str):
    payload = request.get_json(silent=True) or {}
    result = execute_flow(
        flow_id=flow_id,
        values=payload.get("values"),
        context=payload.get("context"),
        step_id=payload.get("step_id")
    )
    return jsonify(result.model_dump())


@api.get("/api/admin/flows")
def admin_flows():
    items = [
        {
            "id": flow.id,
            "name": flow.name,
            "slug": flow.slug,
            "description": flow.description,
            "file_path": flow.file_path,
            "is_active": flow.is_active,
            "created_at": flow.created_at.isoformat() if flow.created_at else None,
            "updated_at": flow.updated_at.isoformat() if flow.updated_at else None,
        }
        for flow in Flow.query.order_by(Flow.name.asc()).all()
    ]
    return jsonify({"items": items, "count": len(items)})


@api.get("/api/admin/flows/<flow_id>")
def admin_flow_detail(flow_id: str):
    flow = Flow.query.filter((Flow.id == flow_id) | (Flow.slug == flow_id)).first()
    if not flow:
      return jsonify({"error": "Flow not found"}), 404
    return jsonify(
        {
            "id": flow.id,
            "name": flow.name,
            "slug": flow.slug,
            "description": flow.description,
            "content_json": flow.content_json,
            "file_path": flow.file_path,
            "is_active": flow.is_active,
            "created_at": flow.created_at.isoformat() if flow.created_at else None,
            "updated_at": flow.updated_at.isoformat() if flow.updated_at else None,
        }
    )


@api.post("/api/admin/flows")
def admin_flow_create():
    payload = request.get_json(silent=True) or {}
    try:
        flow = create_flow(
            name=(payload.get("name") or "").strip(),
            description=(payload.get("description") or "").strip() or None,
            content_json=payload.get("content_json") or "{}",
            is_active=bool(payload.get("is_active", True)),
        )
        return jsonify({"item": {"id": flow.id, "slug": flow.slug}}), 201
    except InvalidFlowError as exc:
        return jsonify({"error": str(exc)}), 400


@api.put("/api/admin/flows/<flow_id>")
def admin_flow_update(flow_id: str):
    payload = request.get_json(silent=True) or {}
    try:
        flow = update_flow(
            flow_id,
            name=(payload.get("name") or "").strip(),
            description=(payload.get("description") or "").strip() or None,
            content_json=payload.get("content_json") or "{}",
            is_active=bool(payload.get("is_active", True)),
        )
        return jsonify({"item": {"id": flow.id, "slug": flow.slug}})
    except FlowNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except InvalidFlowError as exc:
        return jsonify({"error": str(exc)}), 400


@api.delete("/api/admin/flows/<flow_id>")
def admin_flow_delete(flow_id: str):
    try:
        delete_flow(flow_id)
        return jsonify({"ok": True})
    except FlowNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404


@api.get("/api/organizations")
def list_organizations():
    items = [
        {
            "id": organization.id,
            "name": organization.name,
            "slug": organization.slug,
            "code": organization.code,
            "is_active": organization.is_active,
            "created_at": organization.created_at.isoformat() if organization.created_at else None,
            "updated_at": organization.updated_at.isoformat() if organization.updated_at else None,
        }
        for organization in Organization.query.order_by(Organization.name.asc()).all()
    ]
    return jsonify({"items": items, "count": len(items)})


@api.get("/api/admin/organizations")
def admin_organizations():
    return list_organizations()

@api.get("/api/organizations/<organization_id>")
def get_organization(organization_id: str):
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401

    role_name = current_user.role.name if current_user.role else ""
    if role_name != "superadmin" and current_user.organization_id != organization_id:
        return jsonify({"error": "Forbidden"}), 403

    organization = Organization.query.get_or_404(organization_id)
    return _serialize_organization_detail(organization)


@api.get("/api/admin/organizations/<organization_id>")
def admin_organization_detail(organization_id: str):
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
    role_name = current_user.role.name if current_user.role else ""
    if role_name != "superadmin":
        return jsonify({"error": "Forbidden"}), 403

    organization = Organization.query.get_or_404(organization_id)
    return _serialize_organization_detail(organization)


@api.post("/api/admin/organizations/<organization_id>/flows")
def admin_organization_add_flow(organization_id: str):
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
    role_name = current_user.role.name if current_user.role else ""
    if role_name != "superadmin":
        return jsonify({"error": "Forbidden"}), 403

    data = request.get_json(silent=True) or {}
    flow_id = (data.get("flow_id") or "").strip()
    if not flow_id:
        return jsonify({"error": "flow_id is required."}), 400

    organization = Organization.query.get_or_404(organization_id)
    flow = Flow.query.get_or_404(flow_id)
    existing = OrganizationFlowAccess.query.filter_by(
        organization_id=organization.id,
        flow_id=flow.id,
    ).first()
    if existing:
        return jsonify({"ok": True, "message": "Flow already assigned."})

    access = OrganizationFlowAccess(
        organization_id=organization.id,
        flow_id=flow.id,
    )
    db.session.add(access)
    db.session.commit()
    return jsonify({"ok": True, "message": "Flow assigned."}), 201


@api.delete("/api/admin/organizations/<organization_id>/flows/<flow_id>")
def admin_organization_remove_flow(organization_id: str, flow_id: str):
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
    role_name = current_user.role.name if current_user.role else ""
    if role_name != "superadmin":
        return jsonify({"error": "Forbidden"}), 403

    access = OrganizationFlowAccess.query.filter_by(
        organization_id=organization_id,
        flow_id=flow_id,
    ).first()
    if not access:
        return jsonify({"error": "Assignment not found"}), 404
    db.session.delete(access)
    db.session.commit()
    return jsonify({"ok": True})

@api.post("/api/admin/organizations")
def admin_create_organization():
    current_user = _get_current_user()
    if not current_user:    
        return jsonify({"error": "Unauthorized"}), 401
    role_name = current_user.role.name if current_user.role else ""
    if role_name != "superadmin":
        return jsonify({"error": "Forbidden"}), 403
    return _create_organization_from_request()


def _create_organization_from_request():
    data = request.get_json(silent=True) or {}
    organization_name = (data.get("name") or "").strip()
    organization_code = (data.get("code") or "").strip()
    organization_slug = (data.get("slug") or "").strip()
    admin = data.get("admin") or {}
    admin_name = (admin.get("name") or "").strip()
    admin_email = (admin.get("email") or "").strip().lower()
    admin_password = admin.get("password") or ""
    admin_role_name = (admin.get("role") or "admin").strip()

    if not organization_name:
        return jsonify({"error": "Organization name is required."}), 400
    if not organization_code:
        return jsonify({"error": "Organization code is required."}), 400
    if not admin_name:
        return jsonify({"error": "Admin name is required."}), 400
    if not admin_email:
        return jsonify({"error": "Admin email is required."}), 400
    if not admin_password:
        return jsonify({"error": "Admin password is required."}), 400

    if Organization.query.filter_by(slug=organization_code.lower()).first():
        return jsonify({"error": "Organization code already exists."}), 409
    if Organization.query.filter_by(code=organization_code).first():
        return jsonify({"error": "Organization code already exists."}), 409
    if User.query.filter_by(email=admin_email).first():
        return jsonify({"error": "Admin email already exists."}), 409

    role = Role.query.filter_by(name=admin_role_name).first()
    if not role:
        return jsonify({"error": f"Role '{admin_role_name}' not found."}), 400

    organization = Organization(
        name=organization_name,
        slug=organization_slug.lower(),
        code=organization_code,
        is_active=True,
    )
    db.session.add(organization)
    db.session.flush()

    admin_user = User(
        organization_id=organization.id,
        role_id=role.id,
        name=admin_name,
        email=admin_email,
        password_hash=generate_password_hash(admin_password),
        is_active=True,
    )
    db.session.add(admin_user)
    db.session.commit()

    return jsonify(
        {
            "organization": {
                "id": organization.id,
                "name": organization.name,
                "slug": organization.slug,
                "code": organization.code,
                "is_active": organization.is_active,
            },
            "admin": {
                "id": admin_user.id,
                "name": admin_user.name,
                "email": admin_user.email,
                "role": admin_user.role.name if admin_user.role else None,
                "organization_id": admin_user.organization_id,
            },
        }
    ), 201


def _serialize_organization_detail(organization: Organization):
    return jsonify(
        {
            "id": organization.id,
            "name": organization.name,
            "slug": organization.slug,
            "code": organization.code,
            "is_active": organization.is_active,
            "created_at": organization.created_at.isoformat() if organization.created_at else None,
            "updated_at": organization.updated_at.isoformat() if organization.updated_at else None,
            "users": [
                {
                    "id": user.id,
                    "name": user.name,
                    "email": user.email,
                    "role": user.role.name if user.role else None,
                    "is_active": user.is_active,
                }
                for user in organization.users
            ],
            "flows": [
                {
                    "flow_id": access.flow_id,
                    "flow_name": access.flow.name if access.flow else None,
                    "flow_slug": access.flow.slug if access.flow else None,
                }
                for access in organization.flows
            ],
        }
    )


def _find_invitation_by_token(token: str) -> Invitation | None:
    token_hash = sha256(token.encode("utf-8")).hexdigest()
    return Invitation.query.filter_by(token_hash=token_hash, accepted_at=None).first()


@api.get("/api/invitations/validate")
def validate_invitation():
    token = (request.args.get("token") or "").strip()
    if not token:
        return jsonify({"error": "token is required"}), 400

    invitation = _find_invitation_by_token(token)
    if not invitation:
        return jsonify({"error": "Invalid or expired invitation"}), 404

    if invitation.expires_at < _utcnow():
        return jsonify({"error": "Invitation expired"}), 410

    return jsonify(
        {
            "invitation_id": invitation.id,
            "email": invitation.email,
            "role": invitation.role.name if invitation.role else None,
            "organization_id": invitation.organization_id,
            "organization_name": invitation.organization.name if invitation.organization else None,
            "expires_at": invitation.expires_at.isoformat(),
        }
    )


@api.post("/api/invitations/accept")
def accept_invitation():
    data = request.get_json(silent=True) or {}
    token = (data.get("token") or "").strip()
    name = (data.get("name") or "").strip()
    password = data.get("password") or ""

    if not token:
        return jsonify({"error": "token is required"}), 400
    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not password:
        return jsonify({"error": "Password is required."}), 400

    invitation = _find_invitation_by_token(token)
    if not invitation:
        return jsonify({"error": "Invalid or expired invitation"}), 404

    if invitation.expires_at < _utcnow():
        return jsonify({"error": "Invitation expired"}), 410

    existing_user = User.query.filter_by(email=invitation.email).first()
    if existing_user:
        return jsonify({"error": "A user already exists for this invitation email."}), 409

    role = invitation.role
    if not role:
        return jsonify({"error": "Invitation role not found."}), 400

    user = User(
        organization_id=invitation.organization_id,
        role_id=role.id,
        name=name,
        email=invitation.email,
        password_hash=generate_password_hash(password),
        is_active=True,
    )
    invitation.accepted_at = _utcnow()
    db.session.add(user)
    db.session.commit()
    return jsonify(
        {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role.name if user.role else None,
            "organization_id": user.organization_id,
        }
    ), 201





@api.get("/api/users")
def list_users():
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
        
    query = User.query
    role_name = current_user.role.name if current_user.role else ""
    if role_name != "superadmin":
        if role_name == "admin":
            query = query.filter_by(organization_id=current_user.organization_id)
        else:
            return jsonify({"error": "Forbidden"}), 403

    items = [
        {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role.name if user.role else None,
            "organization_id": user.organization_id,
            "organization_name": user.organization.name if user.organization else None,
            "is_active": user.is_active,
            "status": "active" if user.is_active else "inactive",
            "date_joined": user.created_at.isoformat() if user.created_at else None,
        }
        for user in query.order_by(User.name.asc()).all()
    ]
    invitations = []
    if role_name in {"admin", "superadmin"}:
        inv_query = Invitation.query
        if role_name == "admin":
            inv_query = inv_query.filter_by(organization_id=current_user.organization_id)
        invitations = [
            {
                "id": invitation.id,
                "name": None,
                "email": invitation.email,
                "role": invitation.role.name if invitation.role else None,
                "organization_id": invitation.organization_id,
                "organization_name": invitation.organization.name if invitation.organization else None,
                "status": "pending",
                "date_joined": None,
                "created_at": invitation.created_at.isoformat() if invitation.created_at else None,
                "expires_at": invitation.expires_at.isoformat() if invitation.expires_at else None,
            }
            for invitation in inv_query.filter_by(accepted_at=None).order_by(Invitation.created_at.desc()).all()
        ]
    combined = sorted(
        items + invitations,
        key=lambda row: row.get("date_joined") or row.get("created_at") or "",
        reverse=True,
    )
    return jsonify({"items": combined, "count": len(combined)})


@api.post("/api/invitations")
def create_invitation():
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
    role_name = current_user.role.name if current_user.role else ""
    if role_name not in {"admin", "superadmin"}:
        return jsonify({"error": "Forbidden"}), 403

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    invite_role_name = (data.get("role") or "member").strip()
    organization_id = current_user.organization_id

    if not organization_id:
        return jsonify({"error": "Organization not found."}), 400
    if not email:
        return jsonify({"error": "Email is required."}), 400

    if invite_role_name == "user":
        invite_role_name = "member"

    role = Role.query.filter_by(name=invite_role_name).first()
    if not role:
        return jsonify({"error": f"Role '{invite_role_name}' not found."}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user and existing_user.organization_id == organization_id:
        return jsonify({"error": "User already exists in this organization."}), 409

    pending = Invitation.query.filter_by(
        organization_id=organization_id,
        email=email,
        accepted_at=None,
    ).first()
    if pending:
        token = uuid4().hex
        pending.token_hash = sha256(token.encode("utf-8")).hexdigest()
        pending.expires_at = _utcnow() + timedelta(days=7)
        db.session.commit()
        return jsonify({
            "ok": True,
            "message": "Pending invitation already existed. A fresh invitation was issued.",
            "invitation_id": pending.id,
            "registration_link": _build_invitation_link(token),
        }), 200

    token = uuid4().hex
    invitation = Invitation(
        organization_id=organization_id,
        created_by=current_user.id,
        email=email,
        role_id=role.id,
        token_hash=sha256(token.encode("utf-8")).hexdigest(),
        expires_at=_utcnow() + timedelta(days=7),
    )
    db.session.add(invitation)
    db.session.commit()
    return jsonify({
        "ok": True,
        "message": "Invitation created.",
        "invitation_id": invitation.id,
        "registration_link": _build_invitation_link(token),
    }), 201


@api.post("/api/invitations/<invitation_id>/resend")
def resend_invitation(invitation_id: str):
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
    invitation = Invitation.query.get_or_404(invitation_id)
    if current_user.role and current_user.role.name == "admin" and invitation.organization_id != current_user.organization_id:
        return jsonify({"error": "Forbidden"}), 403
    token = uuid4().hex
    invitation.token_hash = sha256(token.encode("utf-8")).hexdigest()
    invitation.expires_at = _utcnow() + timedelta(days=7)
    db.session.commit()
    return jsonify({
        "ok": True,
        "registration_link": _build_invitation_link(token),
    })


@api.delete("/api/invitations/<invitation_id>")
def cancel_invitation(invitation_id: str):
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
    invitation = Invitation.query.get_or_404(invitation_id)
    if current_user.role and current_user.role.name == "admin" and invitation.organization_id != current_user.organization_id:
        return jsonify({"error": "Forbidden"}), 403
    db.session.delete(invitation)
    db.session.commit()
    return jsonify({"ok": True})


@api.put("/api/users/<user_id>")
def update_user(user_id: str):
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
    target = User.query.get_or_404(user_id)
    role_name = current_user.role.name if current_user.role else ""
    if role_name == "admin" and target.organization_id != current_user.organization_id:
        return jsonify({"error": "Forbidden"}), 403

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or target.name).strip()
    new_role_name = (data.get("role") or (target.role.name if target.role else "member")).strip()
    if new_role_name == "user":
        new_role_name = "member"
    new_role = Role.query.filter_by(name=new_role_name).first()
    if not new_role:
        return jsonify({"error": f"Role '{new_role_name}' not found."}), 400

    active_admins = User.query.join(Role).filter(
        User.organization_id == target.organization_id,
        User.is_active.is_(True),
        Role.name == "admin",
    ).count()
    target_is_only_admin = target.role and target.role.name == "admin" and active_admins <= 1
    if target_is_only_admin and new_role.name != "admin":
        return jsonify({"error": "Cannot change the only remaining administrator."}), 400

    target.name = name
    target.role_id = new_role.id
    db.session.commit()
    return jsonify({
        "id": target.id,
        "name": target.name,
        "email": target.email,
        "role": target.role.name if target.role else None,
        "is_active": target.is_active,
    })


@api.delete("/api/users/<user_id>")
def deactivate_user(user_id: str):
    current_user = _get_current_user()
    if not current_user:
        return jsonify({"error": "Unauthorized"}), 401
    target = User.query.get_or_404(user_id)
    role_name = current_user.role.name if current_user.role else ""
    if role_name == "admin" and target.organization_id != current_user.organization_id:
        return jsonify({"error": "Forbidden"}), 403

    active_admins = User.query.join(Role).filter(
        User.organization_id == target.organization_id,
        User.is_active.is_(True),
        Role.name == "admin",
    ).count()
    if target.role and target.role.name == "admin" and active_admins <= 1:
        return jsonify({"error": "Cannot delete the only remaining administrator."}), 400

    target.is_active = False
    db.session.commit()
    return jsonify({"ok": True})


@api.get("/api/profile")
def profile():
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(
        {
            "item": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "role": user.role.name if user.role else None,
                "organization_id": user.organization_id,
                "organization_name": user.organization.name if user.organization else None,
                "is_active": user.is_active,
            }
        }
    )



@api.get("/api/admin/manage-flows")
def manage_flows():
    items = [
        {
            "organization_id": access.organization_id,
            "organization_name": access.organization.name if access.organization else None,
            "flow_id": access.flow_id,
            "flow_name": access.flow.name if access.flow else None,
        }
        for access in OrganizationFlowAccess.query.all()
    ]
    return jsonify({"items": items, "count": len(items)})
