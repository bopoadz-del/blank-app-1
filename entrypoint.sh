#!/usr/bin/env bash
set -e
python scripts/run_migrations.py || true
python scripts/seed_formulas.py || true
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
