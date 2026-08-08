from werkzeug.security import generate_password_hash

from app import create_app
from extensions import db
import json
from pathlib import Path

from models import (
    Flow,
    Form,
    FlowFormStep,
    Organization,
    OrganizationFlowAccess,
    Role,
    User,
)

def get_or_create_role(name, description):
    role = Role.query.filter_by(name=name).first()

    if role is None:
        role = Role(
            name=name,
            description=description,
            is_system=True,
        )
        db.session.add(role)
        db.session.flush()

    return role

def get_or_create_organization(name, slug, code):
    organization = Organization.query.filter_by(slug=slug).first()

    if organization is None:
        organization = Organization(
            name=name,
            slug=slug,
            code=code,
        )
        db.session.add(organization)
        db.session.flush()

    return organization


def get_or_create_user(
    organization_id,
    role_id,
    name,
    email,
    password,
):
    user = User.query.filter_by(email=email).first()

    if user is None:
        user = User(
            organization_id=organization_id,
            role_id=role_id,
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.flush()

    return user


def get_or_create_flow(
    name,
    slug,
    description,
    file_path,
):
    flow = Flow.query.filter_by(slug=slug).first()

    if flow is None:
        flow = Flow(
            name=name,
            slug=slug,
            description=description,
            file_path=file_path,
        )
        db.session.add(flow)
        db.session.flush()

    return flow

def get_or_create_form(
    name,
    slug,
    description,
    content_json,
):
    form = Form.query.filter_by(slug=slug).first()

    if form is None:
        form = Form(
            name=name,
            slug=slug,
            description=description,
            content_json=json.dumps(content_json, indent=2),
            file_path="",
        )

        db.session.add(form)
        db.session.flush()

    return form


def upsert_form_from_file(form_file: Path):
    payload = json.loads(form_file.read_text(encoding="utf-8"))
    form = Form.query.filter_by(slug=payload["id"]).first()

    if form is None:
        form = Form(
            name=payload["name"],
            slug=payload["id"],
            description=payload.get("description"),
            content_json=json.dumps(payload, indent=2),
            file_path=str(form_file.relative_to(form_file.parents[1])),
        )
        db.session.add(form)
        db.session.flush()
        return form

    form.name = payload["name"]
    form.description = payload.get("description")
    form.content_json = json.dumps(payload, indent=2)
    form.file_path = str(form_file.relative_to(form_file.parents[1]))
    return form

def add_form_step(
    flow,
    form,
    step_number,
):
    step = FlowFormStep.query.filter_by(
        flow_id=flow.id,
        step_number=step_number,
    ).first()

    if step is None:
        step = FlowFormStep(
            flow_id=flow.id,
            form_id=form.id,
            step_number=step_number,
            is_required=True,
        )

        db.session.add(step)
    else:
        step.form_id = form.id
        step.is_required = True

    return step


def grant_flow_access(
    organization_id,
    flow_id,
):
    access = OrganizationFlowAccess.query.filter_by(
        organization_id=organization_id,
        flow_id=flow_id,
    ).first()

    if access is None:
        access = OrganizationFlowAccess(
            organization_id=organization_id,
            flow_id=flow_id,
        )
        db.session.add(access)

    return access

def seed():
    # Roles

    admin_role = get_or_create_role(
        "admin",
        "Full access"
    )
    member_role = get_or_create_role(
        "member",
        "Standard access"
    )
    superadmin_role = get_or_create_role(
        "superadmin",
        "Platform access"
    )

    superadmin_user = get_or_create_user(
    None, 
    superadmin_role.id,
    "Superadmin",
    "superadmin@example.com",
    "password123",
    )

    # Organization
    org = get_or_create_organization(
    "Acme Corp",
    "acme-corp",
    "ACME01",
    )
    # Users
    admin_user = get_or_create_user(
        org.id,
        admin_role.id,
        "Alice Admin",
        "alice@acme.com",
        "password123",
    )

    member_user = get_or_create_user(
        org.id,
        member_role.id,
        "Manny Member",
        "member@acme.com",
        "password123",
    )

    forms_dir = Path(__file__).resolve().parent / "forms"
    customer_summary = upsert_form_from_file(forms_dir / "customer_summary.form.json")
    sales_email = upsert_form_from_file(forms_dir / "sales_email.form.json")
    travel_itinerary = upsert_form_from_file(forms_dir / "travel_itinerary.form.json")
    destination_summary = upsert_form_from_file(forms_dir / "destination_summary.form.json")
    travel_checklist = upsert_form_from_file(forms_dir / "travel_checklist.form.json")
    meeting_summary = upsert_form_from_file(forms_dir / "meeting_summary.form.json")
    recipe_creator = upsert_form_from_file(forms_dir / "recipe_creator.form.json")

    client_assessment = get_or_create_flow(
        "Client Assessment",
        "client-assessment",
        "Generate a client summary followed by a personalized sales email.",
        "flows/client_assessment.flow.json",
    )
    travel_planner = get_or_create_flow(
        "Travel Planner",
        "travel-planner",
        "Generate personalized travel itineraries.",
        "",
    )
    recipe_maker = get_or_create_flow(
        "Recipe Maker",
        "recipe-maker",
        "Generate a recipe from ingredients and preferences.",
        "",
    )

    for flow in [client_assessment, travel_planner, recipe_maker]:
        grant_flow_access(org.id, flow.id)

    add_form_step(client_assessment, customer_summary, 1)
    add_form_step(client_assessment, sales_email, 2)

    add_form_step(travel_planner, destination_summary, 1)
    add_form_step(travel_planner, travel_itinerary, 2)
    add_form_step(travel_planner, travel_checklist, 3)

    add_form_step(recipe_maker, recipe_creator, 1)
    add_form_step(recipe_maker, meeting_summary, 2)



    db.session.commit()
    print("Seeding complete.")
    print("Login accounts:")
    print("  admin      alice@acme.com / password123")
    print("  member     member@acme.com / password123")
    print("  superadmin superadmin@example.com / password123")


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        seed()
