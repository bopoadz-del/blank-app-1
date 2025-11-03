from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.deps import get_db
from core.credibility.review_repo import ReviewRepo
router=APIRouter()
@router.get('/pending')
def pending(db: Session = Depends(get_db)): return ReviewRepo(db).list_pending()
@router.post('/{rid}/approve')
def approve(rid:int, db: Session = Depends(get_db)):
    ReviewRepo(db).set_status(rid, 'APPROVED'); return {'ok': True}
@router.post('/{rid}/reject')
def reject(rid:int, db: Session = Depends(get_db)):
    ReviewRepo(db).set_status(rid, 'REJECTED'); return {'ok': True}
