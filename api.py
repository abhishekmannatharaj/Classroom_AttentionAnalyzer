from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import json
import os
import signal

app = FastAPI(title="Attention Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE_PROCESS = None


@app.post("/start-session")
def start_session():
    global PIPELINE_PROCESS

    # kill previous process if still running
    if PIPELINE_PROCESS is not None:
        PIPELINE_PROCESS.terminate()
        PIPELINE_PROCESS = None

    # start realtime pipeline
    PIPELINE_PROCESS = subprocess.Popen(
        ["python", "realtime_pipeline.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return {"status": "pipeline_started"}


@app.get("/engagement")
def engagement():
    if not os.path.exists("output.json"):
        return {"detail": "output.json not found"}

    with open("output.json", "r") as f:
        data = json.load(f)

    return data


@app.post("/end-session")
def end_session():
    global PIPELINE_PROCESS

    if PIPELINE_PROCESS is not None:
        PIPELINE_PROCESS.terminate()
        PIPELINE_PROCESS = None

    return {"status": "pipeline_stopped"}
