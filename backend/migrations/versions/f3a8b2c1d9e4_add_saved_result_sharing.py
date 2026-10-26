"""add saved_result sharing fields

Revision ID: f3a8b2c1d9e4
Revises: cea3f9190e5d
Create Date: 2026-10-25 15:11:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f3a8b2c1d9e4'
down_revision = 'cea3f9190e5d'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('saved_results', sa.Column('is_shared', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('saved_results', sa.Column('share_token', sa.String(length=64), nullable=True))
    op.add_column('saved_results', sa.Column('shared_at', sa.DateTime(timezone=True), nullable=True))
    op.create_index(op.f('ix_saved_results_share_token'), 'saved_results', ['share_token'], unique=True)


def downgrade():
    op.drop_index(op.f('ix_saved_results_share_token'), table_name='saved_results')
    op.drop_column('saved_results', 'shared_at')
    op.drop_column('saved_results', 'share_token')
    op.drop_column('saved_results', 'is_shared')
