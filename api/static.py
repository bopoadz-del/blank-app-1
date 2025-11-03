from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

def mount_static(app: FastAPI):
    app.mount('/static', StaticFiles(directory='ui/web'), name='static')
