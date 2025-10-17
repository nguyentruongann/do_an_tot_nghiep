from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.registry import get_app_config

cfg = get_app_config()
DB_PATH = cfg["app"]["db_path"]
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
