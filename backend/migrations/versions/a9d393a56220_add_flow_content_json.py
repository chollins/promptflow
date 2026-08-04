"""add flow content json

Revision ID: a9d393a56220
Revises: 7d4f9e2a1c8b
Create Date: 2026-07-30 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "a9d393a56220"
down_revision = "7d4f9e2a1c8b"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("flows", schema=None) as batch_op:
        batch_op.add_column(sa.Column("content_json", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("flows", schema=None) as batch_op:
        batch_op.drop_column("content_json")
