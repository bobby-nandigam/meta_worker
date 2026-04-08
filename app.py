#!/usr/bin/env python3
"""FastAPI application for OpenEnv environment."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from openenv import OpenEnv, TaskType, Action

# Initialize FastAPI app
app = FastAPI(
    title="MetaOpenEnv API",
    description="Real-world task simulation environment API",
    version="1.0.0",
)

# Add CORS middleware for HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = OpenEnv()


# Request/Response models
class ResetRequest(BaseModel):
    """Request model for reset endpoint."""
    task_type: Optional[str] = "email_triage"
    task_id: Optional[str] = None


class ActionRequest(BaseModel):
    """Request model for step endpoint."""
    action_type: str
    target_id: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    classification: Optional[str] = None


class ResetResponse(BaseModel):
    """Response model for reset endpoint."""
    status: str
    task_id: str
    task_type: str
    task_name: str
    observation: Dict[str, Any]
    message: str


class StepResponse(BaseModel):
    """Response model for step endpoint."""
    status: str
    observation: Dict[str, Any]
    reward: Dict[str, Any]
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    """Response model for state endpoint."""
    status: str
    state: Dict[str, Any]


class ValidationResponse(BaseModel):
    """Response model for validation endpoint."""
    status: str
    valid: bool
    message: str


# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "OpenEnv API is running"}


# Index endpoint with simple UI
@app.get("/")
def index():
    """Root endpoint with API documentation."""
    return {
        "name": "MetaOpenEnv API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "POST /reset": "Reset environment with task type (email_triage, code_review, content_moderation)",
            "POST /step": "Execute action in current environment",
            "GET /state": "Get current environment state",
            "GET /health": "Health check",
            "GET /config": "Get environment configuration",
            "POST /evaluate": "Evaluate completed episode",
            "GET /test": "Quick test endpoint",
        }
    }


# Test endpoint
@app.get("/test")
def test():
    """Quick test endpoint."""
    try:
        obs = env.reset(TaskType.EMAIL_TRIAGE)
        return {
            "status": "ok",
            "message": "Environment working",
            "task": obs.task_name,
            "test": "Use /reset and /step endpoints with POST requests"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Reset endpoint
@app.post("/reset", response_model=ResetResponse)
def reset(request: Optional[ResetRequest] = None):
    """
    Reset the environment and start a new episode.
    
    Args:
        request: ResetRequest with task_type and optional task_id
        
    Returns:
        ResetResponse with initial observation
    """
    try:
        # Use default if no request provided
        if request is None:
            request = ResetRequest()
        
        # Map string task type to TaskType enum
        task_type_map = {
            "email_triage": TaskType.EMAIL_TRIAGE,
            "code_review": TaskType.CODE_REVIEW,
            "content_moderation": TaskType.CONTENT_MODERATION,
        }
        
        task_type = task_type_map.get(request.task_type or "email_triage")
        if not task_type:
            raise ValueError(f"Unknown task type: {request.task_type}")
        
        # Reset environment
        observation = env.reset(task_type=task_type, task_id=request.task_id)
        
        return ResetResponse(
            status="success",
            task_id=observation.task_id,
            task_type=observation.task_type,
            task_name=observation.task_name,
            observation=observation.model_dump(),
            message=f"Environment reset with task: {observation.task_name}",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Step endpoint
@app.post("/step", response_model=StepResponse)
def step(request: Optional[ActionRequest] = None):
    """
    Execute one step in the environment.
    
    Args:
        request: ActionRequest with action details
        
    Returns:
        StepResponse with observation, reward, and done flag
    """
    try:
        if env.current_task is None:
            raise RuntimeError("Environment not initialized. Call POST /reset first.")
        
        if request is None:
            raise ValueError("Request body required. Provide action_type and other fields.")
        
        # Create Action object
        action = Action(
            action_type=request.action_type,
            target_id=request.target_id,
            confidence=request.confidence,
            reasoning=request.reasoning,
            classification=request.classification,
        )
        
        # Execute step
        step_result = env.step(action)
        
        return StepResponse(
            status="success",
            observation=step_result.observation.model_dump(),
            reward=step_result.reward.model_dump(),
            done=step_result.done,
            info=step_result.info,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# State endpoint
@app.get("/state", response_model=StateResponse)
def get_state():
    """
    Get the current state of the environment.
    
    Returns:
        StateResponse with current state
    """
    try:
        state = env.state()
        return StateResponse(
            status="success",
            state=state,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Validate endpoint
@app.get("/validate", response_model=ValidationResponse)
def validate():
    """
    Validate that the environment is properly set up.
    
    Returns:
        ValidationResponse with validation status
    """
    try:
        # Try to reset and step through a simple task
        obs = env.reset(TaskType.EMAIL_TRIAGE)
        
        # Create a simple action
        action = Action(
            action_type="classify",
            target_id="email_0",
            confidence=0.9,
        )
        action.classification = "work"  # type: ignore
        
        # Execute step
        step_result = env.step(action)
        
        # Check results
        valid = (
            obs is not None and
            step_result is not None and
            step_result.reward is not None
        )
        
        return ValidationResponse(
            status="success",
            valid=valid,
            message="Environment validation passed" if valid else "Environment validation failed",
        )
    except Exception as e:
        return ValidationResponse(
            status="error",
            valid=False,
            message=f"Validation error: {str(e)}",
        )


# Evaluate endpoint
@app.post("/evaluate")
def evaluate():
    """
    Evaluate the completed episode.
    
    Returns:
        Evaluation results with final score
    """
    try:
        result = env.evaluate_episode()
        return {"status": "success", "evaluation": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Get config endpoint
@app.get("/config")
def get_config():
    """Get environment configuration."""
    try:
        config = env.get_config()
        return {"status": "success", "config": config}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import os
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
