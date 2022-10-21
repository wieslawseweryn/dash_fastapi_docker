import os
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import FileResponse

from covid_2 import app

server = FastAPI()
@server.get("/download/{name_file}")
def download_file(name_file: str):
    return FileResponse(path=os.getcwd() + "/data/" + name_file, media_type='application/octet-stream', filename=name_file)
server.add_middleware(SessionMiddleware, secret_key="SECRET_KEY")
server.mount("/", WSGIMiddleware(app.server))




