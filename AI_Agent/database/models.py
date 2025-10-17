from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy import Integer, String, Text, DateTime, Float, func

Base = declarative_base()

class LeadScore(Base):
    __tablename__ = "lead_scores"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id = mapped_column(Integer, index=True)
    model_version = mapped_column(String, index=True)
    score = mapped_column(Float)
    grade = mapped_column(String(2))
    rationale = mapped_column(Text)
    created_at = mapped_column(DateTime, server_default=func.current_timestamp())
