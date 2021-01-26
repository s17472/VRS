"""
Benedykt Kościński
"""
# 3rd party
import uvicorn

# Local
import app.core.settings as settings
from app.api import app

if __name__ == "__main__":
    if settings.DEBUG:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, debug=True)
    else:
        uvicorn.run(app, host=settings.HOST, port=settings.PORT)
