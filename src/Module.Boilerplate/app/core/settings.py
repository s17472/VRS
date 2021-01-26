"""
Benedykt Kościński
"""
# 3rd party
from decouple import config

# Server
HOST = config("HOST", default="127.0.0.1")
PORT = config("PORT", default=8000, cast=int)
DEBUG = config("DEBUG", default=True, cast=bool)

# Module
MODULE_FRAME_COUNT = config("MODULE_FRAMES_COUNT", default=1, cast=int)
MODULE_TYPE = config("UPLOAMODULE_TYPED_DIR", default="boilerplate")
MODULE_VERSION = config("MODULE_VERSION", default="0.0.1")
