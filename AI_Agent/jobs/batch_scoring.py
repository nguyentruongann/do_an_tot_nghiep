import argparse
from database.session import SessionLocal
from agents.orchestrator import Orchestrator

def main(data_type: str, limit: int | None):
    db = SessionLocal()
    orch = Orchestrator()
    res = orch.handle_score_batch(db, data_type=data_type, limit=limit)
    print(f"Scored: {len(res)} rows")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_type", type=str, required=True, help="e.g. bank_marketing")
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    main(args.data_type, args.limit)
