"""add_organization_model_labels

Revision ID: 1a2b3c4d5e6f
Revises: f3a8b2c1d9e4
Create Date: 2026-10-26 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = 'f3a8b2c1d9e4'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'organization_model_labels',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=False),
        sa.Column('model_id', sa.String(length=255), nullable=False),
        sa.Column('label', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('organization_id', 'model_id', name='uq_org_model_label'),
    )
    with op.batch_alter_table('organization_model_labels', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_organization_model_labels_organization_id'), ['organization_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_organization_model_labels_model_id'), ['model_id'], unique=False)


def downgrade():
    with op.batch_alter_table('organization_model_labels', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_organization_model_labels_model_id'))
        batch_op.drop_index(batch_op.f('ix_organization_model_labels_organization_id'))

    op.drop_table('organization_model_labels')
