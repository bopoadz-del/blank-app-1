import os
from fastapi import Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

def cors_middleware(app):
    origins=os.getenv('CORS_ALLOW_ORIGINS','*').split(',')
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=True,allow_methods=['*'],allow_headers=['*'])

def require_api_key(x_api_key: str = Header(None)):
    expected=os.getenv('API_KEY','dev-key')
    if x_api_key!=expected:
        raise HTTPException(401,'Invalid API key')
