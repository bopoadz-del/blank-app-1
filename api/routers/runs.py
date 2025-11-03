from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.deps import get_db
from core.pipeline import run_pipeline
from api.metrics import runs_total, decisions_total

router=APIRouter()

@router.post('/runs/submit')
def submit(payload: dict, db: Session = Depends(get_db)):
    rid, result, used = run_pipeline(db, payload.get('formula_code'), payload.get('inputs',{}), payload.get('context_text',''))
    try:
        runs_total.inc(); decisions_total.labels(result['decision']).inc()
    except Exception:
        pass
    return {'run_id': rid, 'formula_code': used, **result}

@router.get('/runs/history')
def history(limit: int = 50, db: Session = Depends(get_db)):
    rows=db.execute('SELECT id, created_at, formula_code, confidence, decision, result_value, result_unit FROM runs ORDER BY id DESC LIMIT :lim', {'lim':limit}).mappings().all()
    return [dict(r) for r in rows]
