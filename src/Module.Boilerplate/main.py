import uvicorn
import shutil
from typing import List

from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile

app = FastAPI()


@app.post("/upload/")
async def upload_files_for_analysis(files: List[UploadFile] = File(...)):
    for f in files:
        with open(f.filename, "wb") as buffer:
            buffer.write(f.file.read())

    return {"uploaded": len(files)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, debug=True)
