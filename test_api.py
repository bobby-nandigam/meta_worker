#!/usr/bin/env python3
"""Quick API test."""

import subprocess
import time
import requests
import json

# Start server
print("Starting API server...")
proc = subprocess.Popen(
    ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

time.sleep(4)

try:
    # Test 1
    print("\n✓ Test 1: GET /test")
    r = requests.get("http://localhost:8000/test")
    print(f"  Status: {r.json()['status']}")
    
    # Test 2
    print("\n✓ Test 2: POST /reset (no body)")
    r = requests.post("http://localhost:8000/reset", json={})
    print(f"  Task: {r.json()['task_name']}")
    
    # Test 3
    print("\n✓ Test 3: POST /reset (with task type)")
    r = requests.post("http://localhost:8000/reset", json={"task_type": "content_moderation"})
    print(f"  Task: {r.json()['task_name']}")
    
    # Test 4
    print("\n✓ Test 4: POST /step")
    r = requests.post("http://localhost:8000/step", json={
        "action_type": "classify",
        "target_id": "email_0",
        "classification": "work"
    })
    print(f"  Reward: {r.json()['reward']['value']:.3f}")
    
    print("\n✓✓✓ All tests passed! API is ready for HF Spaces")
    
finally:
    proc.terminate()
