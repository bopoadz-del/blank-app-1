from sqlalchemy.orm import Session
from sqlalchemy import text
class ReviewRepo:
    def __init__(self, db:Session): self.db=db
    def create_for_run(self, run_id:int)->int:
        row=self.db.execute(text("INSERT INTO reviews(run_id,status) VALUES (:r,'PENDING') RETURNING id"),{'r':run_id}).mappings().first()
        self.db.commit(); return int(row['id'])
    def list_pending(self):
        rows=self.db.execute(text("SELECT r.id as review_id, runs.id as run_id, runs.decision, runs.confidence, runs.formula_code, runs.created_at FROM reviews r JOIN runs ON runs.id=r.run_id WHERE r.status='PENDING' ORDER BY r.id DESC")).mappings().all()
        return [dict(x) for x in rows]
    def set_status(self, review_id:int, status:str, notes:str=None):
        self.db.execute(text("UPDATE reviews SET status=:s, notes=:n, updated_at=NOW() WHERE id=:i"),{'s':status,'n':notes,'i':review_id}); self.db.commit()
