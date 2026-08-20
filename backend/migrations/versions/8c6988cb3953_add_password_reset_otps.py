"""add password_reset_otps

Revision ID: 8c6988cb3953
Revises: b2a7f0d1c901
Create Date: 2026-08-14 10:35:49.784428

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8c6988cb3953'
down_revision = 'b2a7f0d1c901'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'password_reset_otps',
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('otp_hash', sa.String(length=255), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False
        ),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['users.id'],
            ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('password_reset_otps')