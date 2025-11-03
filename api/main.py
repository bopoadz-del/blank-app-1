from fastapi import FastAPI, Depends, Response
from api.middleware import RequestIDMiddleware
from api.security import cors_middleware, require_api_key
from api.metrics import expose_metrics
from api.static import mount_static
from api.routers import runs, ingest, reviews, reports, jobs

app=FastAPI(title='Reasoner V3')
app.add_middleware(RequestIDMiddleware)
cors_middleware(app)
app.include_router(runs.router, dependencies=[Depends(require_api_key)])
app.include_router(ingest.router, prefix='/ingest', dependencies=[Depends(require_api_key)])
app.include_router(reviews.router, prefix='/reviews', dependencies=[Depends(require_api_key)])
app.include_router(reports.router, prefix='/runs', dependencies=[Depends(require_api_key)])
app.include_router(jobs.router, prefix='/jobs', dependencies=[Depends(require_api_key)])

@app.get('/metrics')
def metrics():
    data, ctype = expose_metrics(); return Response(content=data, media_type=ctype)

@app.get('/health')
def health(): return {'ok': True}

mount_static(app)
