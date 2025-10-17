from sqlalchemy.orm import Session
from sqlalchemy import text
from database.models import LeadScore

def save_score(db: Session, customer_id: int, score: float, grade: str, rationale: str, model_version: str):
    existing = db.query(LeadScore).filter(LeadScore.customer_id==customer_id, LeadScore.model_version==model_version).one_or_none()
    if existing:
        db.delete(existing); db.flush()
    rec = LeadScore(customer_id=customer_id, score=score, grade=grade, rationale=rationale, model_version=model_version)
    db.add(rec); db.commit()

def fetch_bank_rows_all(db: Session, limit: int = 100):
    q = text("SELECT * FROM bank_customers LIMIT :limit")
    res = db.execute(q, {"limit": limit})
    cols = res.keys()
    return [dict(zip(cols, row)) for row in res.fetchall()]

def fetch_bank_rows_by_ids(db: Session, ids: list[int]):
    placeholders = ",".join([str(int(x)) for x in ids])
    q = text(f"SELECT * FROM bank_customers WHERE ID IN ({placeholders})")
    res = db.execute(q)
    cols = res.keys()
    return [dict(zip(cols, row)) for row in res.fetchall()]
