#!/usr/bin/env python3
"""Minimal test app to verify FastAPI works on HF Spaces."""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from MetaOpenEnv!", "status": "working"}

@app.get("/test")
def test():
    return {"test": "OK", "port": "7860"}

@app.post("/reset")
def reset(task_type: str = "email_triage"):
    return {"status": "success", "task": task_type}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
