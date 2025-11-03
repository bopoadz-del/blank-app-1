import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
engine=create_engine(os.getenv('DATABASE_URL'), pool_pre_ping=True)
SessionLocal=sessionmaker(bind=engine)

def get_db():
    db:Session=SessionLocal()
    try:
        yield db
    finally:
        db.close()
