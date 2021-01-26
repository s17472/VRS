"""
Benedykt Kościński
"""
# Builtins
import os

# 3rd party
from fastapi import UploadFile


def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as buffer:
            buffer.write(upload_file.file.read())
    finally:
        upload_file.file.close()
