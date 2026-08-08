"""Add forms and flow_form_steps

Revision ID: b2a7f0d1c901
Revises: a9d393a56220
Create Date: 2026-08-05 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b2a7f0d1c901"
down_revision = "a9d393a56220"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "forms",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=150), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("content_json", sa.Text(), nullable=True),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    with op.batch_alter_table("forms", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_forms_slug"), ["slug"], unique=True)

    op.create_table(
        "flow_form_steps",
        sa.Column("flow_id", sa.String(length=36), nullable=False),
        sa.Column("form_id", sa.String(length=36), nullable=False),
        sa.Column("step_number", sa.Integer(), nullable=False),
        sa.Column("is_required", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["flow_id"], ["flows.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["form_id"], ["forms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("flow_id", "form_id"),
        sa.UniqueConstraint("flow_id", "step_number"),
    )


def downgrade():
    op.drop_table("flow_form_steps")
    with op.batch_alter_table("forms", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_forms_slug"))
    op.drop_table("forms")
