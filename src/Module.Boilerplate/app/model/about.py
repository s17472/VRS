"""
Benedykt Kościński
"""
# 3rd party
from pydantic import BaseModel

# Local
import app.core.settings as settings


class AboutResult(BaseModel):
    module_type: str = settings.MODULE_TYPE
    module_version: str = settings.MODULE_VERSION
    module_frame_count: str = settings.MODULE_FRAME_COUNT
