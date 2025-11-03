from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from core.pipeline import run_pipeline
DATABASE_URL=os.getenv('DATABASE_URL')
engine=create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal=sessionmaker(bind=engine)

def async_run(formula_code, inputs, context_text):
    db=SessionLocal()
    try:
        rid,res,used=run_pipeline(db, formula_code, inputs, context_text)
        return {'run_id': rid, 'result': res, 'formula_code': used}
    finally:
        db.close()
