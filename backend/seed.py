from werkzeug.security import generate_password_hash

from app import create_app
from extensions import db
from models import Flow, Organization, OrganizationFlowAccess, Role, User

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

    # Flows
    flow1 = get_or_create_flow(
        "Client Assessment",
        "client-assessment",
        "Generate a client summary followed by a personalized sales email.",
        "flows/client_assessment.flow.json",
    )

    flow2 = get_or_create_flow(
        "Personalized Travel Planner",
        "travel-planner",
        "Generate a destination guide, personalized itinerary, and travel preparation checklist.",
        "flows/travel_planner.flow.json",
    )

    # Org <-> Flow access
    grant_flow_access(
        org.id,
        flow1.id,
    )

    grant_flow_access(
        org.id,
        flow2.id,
    )

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


