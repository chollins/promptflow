"""add_org_settings_and_ownership

Revision ID: 2b3c4d5e6f7a
Revises: 1a2b3c4d5e6f
Create Date: 2026-11-23 15:55:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2b3c4d5e6f7a'
down_revision = '1a2b3c4d5e6f'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('organizations', schema=None) as batch_op:
        batch_op.add_column(sa.Column('auto_next_step', sa.Boolean(), nullable=False, server_default=sa.text('0')))
        batch_op.add_column(sa.Column('allow_user_create_flows', sa.Boolean(), nullable=False, server_default=sa.text('0')))
        batch_op.add_column(sa.Column('allow_user_share_flows', sa.Boolean(), nullable=False, server_default=sa.text('0')))

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.add_column(sa.Column('can_create_flows', sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column('can_share_flows', sa.Boolean(), nullable=True))

    with op.batch_alter_table('flows', schema=None) as batch_op:
        batch_op.add_column(sa.Column('organization_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('auto_next_step', sa.Boolean(), nullable=True))
        batch_op.create_foreign_key('fk_flows_organization_id', 'organizations', ['organization_id'], ['id'], ondelete='CASCADE')

    with op.batch_alter_table('forms', schema=None) as batch_op:
        batch_op.add_column(sa.Column('organization_id', sa.String(length=36), nullable=True))
        batch_op.create_foreign_key('fk_forms_organization_id', 'organizations', ['organization_id'], ['id'], ondelete='CASCADE')


def downgrade():
    with op.batch_alter_table('forms', schema=None) as batch_op:
        batch_op.drop_constraint('fk_forms_organization_id', type_='foreignkey')
        batch_op.drop_column('organization_id')

    with op.batch_alter_table('flows', schema=None) as batch_op:
        batch_op.drop_constraint('fk_flows_organization_id', type_='foreignkey')
        batch_op.drop_column('auto_next_step')
        batch_op.drop_column('organization_id')

    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_column('can_share_flows')
        batch_op.drop_column('can_create_flows')

    with op.batch_alter_table('organizations', schema=None) as batch_op:
        batch_op.drop_column('allow_user_share_flows')
        batch_op.drop_column('allow_user_create_flows')
        batch_op.drop_column('auto_next_step')
