from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import subprocess

#py -m  uvicorn proyectoAPI:app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message":"hello!"}

@app.get("/run-script")
def run_script():
    result = subprocess.run(["py", "proyectoImagenes.py"], capture_output=True, text=True)
    return {"output": result.stdout, "error": result.stderr}