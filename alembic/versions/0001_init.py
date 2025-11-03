from alembic import op
import sqlalchemy as sa

revision = '0001_init'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('formulas',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('code', sa.Text, nullable=False, unique=True),
        sa.Column('name', sa.Text, nullable=False),
        sa.Column('expression', sa.Text, nullable=False),
        sa.Column('inputs', sa.JSON, nullable=False),
        sa.Column('output_unit', sa.Text, nullable=False),
        sa.Column('domain', sa.Text),
        sa.Column('version', sa.Integer, nullable=False, server_default='1'),
        sa.Column('checksum', sa.Text),
        sa.Column('target', sa.Text),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
    )
    op.create_table('runs',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('context', sa.JSON),
        sa.Column('formula_code', sa.Text),
        sa.Column('inputs', sa.JSON),
        sa.Column('result_value', sa.Float),
        sa.Column('result_unit', sa.Text),
        sa.Column('confidence', sa.Float),
        sa.Column('decision', sa.Text),
        sa.Column('validation', sa.JSON),
        sa.Column('lineage', sa.JSON),
    )

def downgrade():
    op.drop_table('runs')
    op.drop_table('formulas')
