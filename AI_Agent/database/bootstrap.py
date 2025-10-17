from database.session import engine
from database.models import Base

def bootstrap_db():
    Base.metadata.create_all(bind=engine)
