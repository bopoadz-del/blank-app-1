from alembic import context
from sqlalchemy import engine_from_config, pool
import os
config = context.config
section = config.config_ini_section
config.set_section_option(section, "DATABASE_URL", os.getenv("DATABASE_URL"))

def run_migrations_offline():
    context.configure(url=os.getenv("DATABASE_URL"), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(config.get_section(config.config_ini_section), prefix="", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
