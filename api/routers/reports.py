from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from api.deps import get_db
from jinja2 import Template
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO
import json
router=APIRouter()

def _load(db:Session, rid:int):
    row=db.execute(text('SELECT * FROM runs WHERE id=:i'),{'i':rid}).mappings().first()
    if not row: raise HTTPException(404,'Run not found')
    return dict(row)

@router.get('/{run_id}/report')
def report(run_id:int, fmt:str=Query('html', enum=['html','pdf']), db: Session = Depends(get_db)):
    run=_load(db, run_id)
    validation=json.loads(run['validation']); lineage=json.loads(run['lineage']); context=json.loads(run['context'])
    normalized=lineage.get('normalized_inputs',{})
    tpl=open('ui/reports/run_report.md.j2','r').read()
    html=Template(tpl).render(run=run, lineage=lineage, normalized=normalized,
                               result_value=run['result_value'], result_unit=run['result_unit'],
                               decision=run['decision'], confidence=run['confidence'],
                               validation=validation, context=context)
    if fmt=='html': return HTMLResponse(f"<pre>{html}</pre>")
    buf=BytesIO(); c=canvas.Canvas(buf, pagesize=A4)
    w,h=A4; y=h-40
    for line in html.splitlines():
        if y<40: c.showPage(); y=h-40
        c.drawString(40,y,line[:110]); y-=14
    c.save(); buf.seek(0)
    return StreamingResponse(buf, media_type='application/pdf', headers={'Content-Disposition': f'attachment; filename=run_{run_id}.pdf'})
