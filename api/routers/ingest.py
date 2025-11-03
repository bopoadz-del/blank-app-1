from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import tempfile, os, json
from ingestion.router import parse_file

MAX_BYTES=int(os.getenv('INGEST_MAX_BYTES','25000000'))
ALLOWED_EXTS={'.csv','.json','.xlsx','.xls'}
router=APIRouter()

def _enforce(f: UploadFile):
    ext=os.path.splitext(f.filename or '')[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(415, f'Unsupported file type: {ext}')

@router.post('/upload')
async def upload(file: UploadFile = File(...), mapping: Optional[str] = Form(None), rows: int = Form(5)):
    _enforce(file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        data=await file.read()
        if len(data)>MAX_BYTES: raise HTTPException(413, f'File too large (> {MAX_BYTES} bytes)')
        tmp.write(data); path=tmp.name
    try:
        parsed=parse_file(path)
    finally:
        os.unlink(path)
    preview=parsed if isinstance(parsed,list) else [parsed]
    preview=preview[:max(1, min(int(rows), 50))]
    mapped=[]; errors=[]
    if mapping:
        m=json.loads(mapping)
        for row in preview:
            out={}; err={}
            for var, col in m.items():
                try:
                    if isinstance(col, dict):
                        colname=col.get('column'); unit=col.get('unit')
                    else:
                        colname=col; unit=None
                    if colname in row and row[colname] not in (None, ''):
                        val=float(row[colname]); out[var]={'value':val, **({'unit':unit} if unit else {})}
                    else:
                        err[var]='missing'
                except Exception as ex:
                    err[var]=f'cast_error: {ex}'
            mapped.append(out); errors.append(err)
    return {'preview': preview, 'mapped': mapped, 'errors': errors}
