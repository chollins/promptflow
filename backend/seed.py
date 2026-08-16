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

def upsert_form(
    name,
    slug,
    description,
    content,
):
    form = Form.query.filter_by(slug=slug).first()

    if form is None:
        form = Form(
            name=name,
            slug=slug,
            description=description,
            content_json=json.dumps(content, indent=2),
            file_path="",
        )
        db.session.add(form)
        db.session.flush()
    else:
        form.name = name
        form.description = description
        form.content_json = json.dumps(content, indent=2)
        form.file_path = ""

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

    # forms_dir = Path(__file__).resolve().parent / "forms"
    # customer_summary = upsert_form_from_file(
    #     forms_dir / "customer_summary.form.json"
    # )
    # sales_email = upsert_form_from_file(
    #     forms_dir / "sales_email.form.json"
    # )
    # travel_itinerary = upsert_form_from_file(
    #     forms_dir / "travel_itinerary.form.json"
    # )
    # destination_summary = upsert_form_from_file(
    #     forms_dir / "destination_summary.form.json"
    # )
    # travel_checklist = upsert_form_from_file(
    #     forms_dir / "travel_checklist.form.json"
    # )
    # meeting_summary = upsert_form_from_file(
    #     forms_dir / "meeting_summary.form.json"
    # )
    # recipe_creator = upsert_form_from_file(
    #     forms_dir / "recipe_creator.form.json"
    # )

    stretch_information = upsert_form(
        "Stretch Information",
        "stretch_information",
        "Collect basic information for generating a personalized stretch guide.",
        {
            "id": "stretch_information",
            "name": "Stretch Information",
            "version": "1.0",
            "fields": [
                {
                    "id": "name",
                    "label": "Name",
                    "type": "text",
                    "required": True,
                    "options": [],
                },
                {
                    "id": "area",
                    "label": "Body Area",
                    "type": "radio",
                    "required": True,
                    "options": [
                        "Arms",
                        "Legs",
                        "Chest",
                        "Abs",
                        "Back",
                        "Shoulders",
                        "Neck",
                    ],
                },
            ],
            "prompt": {
                "system": "You are an expert in human anatomy and stretching.",
                "user": (
                    "List 5 key muscles associated with the human "
                    "{{area}}. Return only a JSON object containing "
                    "a muscles array of strings."
                ),
            },
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "temperature": 0.3,
            },
            "output": {
                "type": "object",
                "schema": {
                    "muscles": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
            },
            "execution": {
                "mode": "interactive",
            },
        },
    )

    select_muscle = upsert_form(
        "Select Muscle",
        "stretch_muscle",
        "Select a muscle generated from the selected body area.",
        {
            "id": "stretch_muscle",
            "name": "Select Muscle",
            "version": "1.0",
            "fields": [
                {
                    "id": "muscle",
                    "label": "Muscle",
                    "type": "radio",
                    "required": True,
                    "options": [],
                    "data_source": {
                        "type": "step_output",
                        "step_id": "stretch_information",
                        "path": "muscles",
                    },
                },
            ],
            "prompt": {
                "system": "You are an expert stretching coach.",
                "user": (
                    "List 5 common stretches specifically targeting "
                    "the {{muscle}}. Return only a JSON object "
                    "containing a stretches array of strings."
                ),
            },
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "temperature": 0.3,
            },
            "output": {
                "type": "object",
                "schema": {
                    "stretches": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
            },
            "execution": {
                "mode": "interactive",
            },
        },
    )

    stretch_guide = upsert_form(
        "Generate Stretch Guide",
        "stretch_guide",
        "Generate a personalized stretching guide from the selected stretches.",
        {
            "id": "stretch_guide",
            "name": "Generate Stretch Guide",
            "version": "1.0",
            "fields": [
                {
                    "id": "stretches",
                    "label": "Select Stretches",
                    "type": "checkbox",
                    "required": True,
                    "options": [],
                    "data_source": {
                        "type": "step_output",
                        "step_id": "select_muscle",
                        "path": "stretches",
                    },
                },
            ],
            "prompt": {
                "system": "You are an expert stretching coach.",
                "user": (
                    "Create a personalized stretching guide for "
                    "{{steps.stretch_information.input.name}}. "
                    "Body area: "
                    "{{steps.stretch_information.input.area}}. "
                    "Target muscle: "
                    "{{steps.select_muscle.input.muscle}}. "
                    "Selected stretches: {{stretches}}. "
                    "For each stretch provide step-by-step "
                    "instructions, benefits, and precautions."
                ),
            },
            "model": {
                "provider": "openai",
                "name": "gpt-4o-mini",
                "temperature": 0.7,
            },
            "output": {
                "type": "object",
                "schema": {
                    "title": {
                        "type": "string",
                    },
                    "introduction": {
                        "type": "string",
                    },
                    "stretches": {
                        "type": "array",
                    },
                    "precautions": {
                        "type": "array",
                    },
                },
            },
            "execution": {
                "mode": "interactive",
            },
        },
    )

    stretch_guide_flow = get_or_create_flow(
    "Stretch Guide",
    "stretch-guide",
    "Generate a personalized stretching guide based on body area, muscle, and selected stretches.",
    "",
    )

    add_form_step(
        stretch_guide_flow,
        stretch_information,
        1,
    )

    add_form_step(
        stretch_guide_flow,
        select_muscle,
        2,
    )

    add_form_step(
        stretch_guide_flow,
        stretch_guide,
        3,
    )


    for flow in [
        stretch_guide_flow,
    ]:
        grant_flow_access(org.id, flow.id)

    # add_form_step(client_assessment, customer_summary, 1)
    # add_form_step(client_assessment, sales_email, 2)

    # add_form_step(travel_planner, destination_summary, 1)
    # add_form_step(travel_planner, travel_itinerary, 2)
    # add_form_step(travel_planner, travel_checklist, 3)

    # add_form_step(recipe_maker, recipe_creator, 1)
    # add_form_step(recipe_maker, meeting_summary, 2)




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
