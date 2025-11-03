from alembic import op
import sqlalchemy as sa

revision = '0003_add_target'
down_revision = '0002_reviews'
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table('formulas') as b:
        b.add_column(sa.Column('target', sa.Text, nullable=True))

def downgrade():
    with op.batch_alter_table('formulas') as b:
        b.drop_column('target')
