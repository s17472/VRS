"""
Benedykt Kościński
"""
# Builtins
import json
import logging
import time

# 3rd party
import seqlog

# Local
import app.core.settings as settings
from app.router.router import router
from fastapi import FastAPI


def get_app() -> FastAPI:
    app = FastAPI(title=settings.MODULE_TYPE, version=settings.MODULE_VERSION)
    app.include_router(router, prefix='/api')
    return app

app = get_app()
