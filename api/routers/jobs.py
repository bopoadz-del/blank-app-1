from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.deps import get_db
from rq import Queue
import redis, os
from core.jobs.runner import async_run
router=APIRouter()
_conn=redis.from_url(os.getenv('REDIS_URL','redis://redis:6379/0'))
_q=Queue('default', connection=_conn)
@router.post('/submit')
def submit(payload:dict, db: Session = Depends(get_db)):
    job=_q.enqueue(async_run, payload.get('formula_code'), payload.get('inputs'), payload.get('context_text',''))
    return {'job_id': job.get_id()}
@router.get('/{job_id}')
def status(job_id:str):
    from rq.job import Job
    job=Job.fetch(job_id, connection=_conn)
    out={'status': job.get_status()}
    if job.result: out['result']=job.result
    return out
