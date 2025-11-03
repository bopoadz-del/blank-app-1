from alembic import op
import sqlalchemy as sa

revision = '0002_reviews'
down_revision = '0001_init'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('reviews',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('run_id', sa.Integer, nullable=False),
        sa.Column('status', sa.Text, nullable=False, server_default='PENDING'),
        sa.Column('notes', sa.Text),
        sa.Column('created_at', sa.DateTime, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime, server_default=sa.text('NOW()'))
    )

def downgrade():
    op.drop_table('reviews')
