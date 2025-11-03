from sqlalchemy.orm import Session
from sqlalchemy import text
DEFAULT_WEIGHT=1.0; ALPHA=0.2
class TinkerTracker:
    def __init__(self, db:Session): self.db=db
    def get_weight(self, code:str)->float:
        row=self.db.execute(text("SELECT weight FROM tinker_stats WHERE formula_code=:c"),{'c':code}).mappings().first()
        return float(row['weight']) if row else DEFAULT_WEIGHT
    def update(self, code:str, success:bool):
        row=self.db.execute(text("SELECT successes,failures,weight FROM tinker_stats WHERE formula_code=:c"),{'c':code}).mappings().first()
        if not row:
            s=1 if success else 0; f=0 if success else 1; w=DEFAULT_WEIGHT+(ALPHA if success else -ALPHA)
            self.db.execute(text("INSERT INTO tinker_stats(formula_code,successes,failures,weight) VALUES (:c,:s,:f,:w)"),{'c':code,'s':s,'f':f,'w':w})
        else:
            s=int(row['successes'])+(1 if success else 0); f=int(row['failures'])+(0 if success else 1)
            w0=float(row['weight']); w=max(0.1, w0+(ALPHA if success else -ALPHA))
            self.db.execute(text("UPDATE tinker_stats SET successes=:s,failures=:f,weight=:w,updated_at=NOW() WHERE formula_code=:c"),{'c':code,'s':s,'f':f,'w':w})
        self.db.commit()
