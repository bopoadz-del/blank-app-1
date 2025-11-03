from alembic import op
import sqlalchemy as sa

revision = '0004_tinker_stats'
down_revision = '0003_add_target'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('tinker_stats',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('formula_code', sa.Text, nullable=False, unique=True),
        sa.Column('successes', sa.Integer, nullable=False, server_default='0'),
        sa.Column('failures', sa.Integer, nullable=False, server_default='0'),
        sa.Column('weight', sa.Float, nullable=False, server_default='1.0'),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('NOW()'))
    )

def downgrade():
    op.drop_table('tinker_stats')
