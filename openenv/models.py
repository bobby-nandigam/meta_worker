"""Pydantic models for OpenEnv specification."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TaskType(str, Enum):
    """Enumeration of task types in the environment."""
    EMAIL_TRIAGE = "email_triage"
    CODE_REVIEW = "code_review"
    CONTENT_MODERATION = "content_moderation"


class Observation(BaseModel):
    """Observation structure returned by the environment."""
    
    task_id: str = Field(description="Unique identifier for the current task")
    task_type: TaskType = Field(description="Type of task being performed")
    task_name: str = Field(description="Human-readable task name")
    data: Dict[str, Any] = Field(description="Task-specific data/content")
    step_count: int = Field(description="Number of steps taken in this episode")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional metadata about the observation"
    )
    
    class Config:
        use_enum_values = True


class Action(BaseModel):
    """Action structure representing agent decisions."""
    
    action_type: str = Field(
        description="Type of action (e.g., 'classify', 'flag', 'approve', 'request_changes')"
    )
    target_id: Optional[str] = Field(
        default=None, 
        description="ID of the item being acted upon"
    )
    confidence: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for the action"
    )
    reasoning: Optional[str] = Field(
        default=None, 
        description="Explanation for the action taken"
    )
    classification: Optional[str] = Field(
        default=None,
        description="Classification label (e.g., 'work', 'spam' for email triage)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional action metadata"
    )


class Reward(BaseModel):
    """Reward structure providing feedback."""
    
    value: float = Field(description="Reward value for this step")
    success: bool = Field(description="Whether the action was successful")
    message: str = Field(description="Human-readable feedback message")
    components: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Breakdown of reward components"
    )


class StepResult(BaseModel):
    """Result of a step() call in the environment."""
    
    observation: Observation = Field(description="New observation after the step")
    reward: Reward = Field(description="Reward for the action")
    done: bool = Field(description="Whether the episode is finished")
    info: Dict[str, Any] = Field(description="Additional info and metadata")


class EnvironmentConfig(BaseModel):
    """Configuration for the OpenEnv environment."""
    
    name: str = Field(description="Environment name")
    version: str = Field(description="Environment version")
    description: str = Field(description="Environment description")
    max_episode_steps: int = Field(default=100, description="Maximum steps per episode")
    action_space_description: str = Field(description="Description of action space")
    observation_space_description: str = Field(description="Description of observation space")
    tasks: List[Dict[str, Any]] = Field(description="List of available tasks")
