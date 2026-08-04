"""Add session_token to users

Revision ID: 7d4f9e2a1c8b
Revises: 191780debf2b
Create Date: 2026-07-30 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "7d4f9e2a1c8b"
down_revision = "191780debf2b"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("session_token", sa.String(length=64), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_users_session_token"),
            ["session_token"],
            unique=True,
        )


def downgrade():
    with op.batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_session_token"))
        batch_op.drop_column("session_token")
