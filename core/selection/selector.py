from sqlalchemy.orm import Session
from sqlalchemy import text
from core.tinker.tracker import TinkerTracker

def _coverage(expected, provided):
    if not expected: return 0.0
    need=set(expected.keys()); have=set([k for k in provided.keys() if provided[k] is not None])
    return len(need & have)/len(need)

def select_candidates(db:Session, domain:str, provided_inputs:dict, limit:int=5):
    rows=db.execute(text('SELECT code,name,domain,inputs FROM formulas')).mappings().all()
    scored=[]
    for r in rows:
        dom=1.0 if domain and r['domain']==domain else (0.5 if domain else 0.4)
        exp=r['inputs'] if isinstance(r['inputs'],dict) else {}
        cov=_coverage(exp, provided_inputs)
        score=(0.6*dom+0.25*cov)*TinkerTracker(db).get_weight(r['code'])
        scored.append((r['code'],score))
    scored.sort(key=lambda x:x[1], reverse=True)
    return [c for c,_ in scored[:limit]]
