"""Core OpenEnv environment implementation."""

import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import copy

from .models import (
    TaskType, Observation, Action, Reward, StepResult, EnvironmentConfig
)
from .tasks import EmailTriageTask, CodeReviewTask, ContentModerationTask, Task
from .graders import EmailTriageGrader, CodeReviewGrader, ContentModerationGrader
from .reward import RewardFunction, RewardConfig


class OpenEnv:
    """
    OpenEnv Environment for Real-World Task Simulation.
    
    Implements the OpenEnv specification with:
    - Typed Observation, Action, Reward models using Pydantic
    - step(action) → returns (observation, reward, done, info)
    - reset() → returns the initial observation
    - state() → returns the current state
    """
    
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        """Initialize the environment."""
        self.config = config or self._default_config()
        self.current_task: Optional[Task] = None
        self.current_task_type: Optional[TaskType] = None
        self.reward_function = RewardFunction(RewardConfig())
        self.trajectory = []
        self.episode_step = 0
        self.episode_reward = 0.0
        
        # Task registry
        self.task_registry = {
            TaskType.EMAIL_TRIAGE: EmailTriageTask,
            TaskType.CODE_REVIEW: CodeReviewTask,
            TaskType.CONTENT_MODERATION: ContentModerationTask,
        }
        
        # Grader registry
        self.grader_registry = {
            TaskType.EMAIL_TRIAGE: EmailTriageGrader(),
            TaskType.CODE_REVIEW: CodeReviewGrader(),
            TaskType.CONTENT_MODERATION: ContentModerationGrader(),
        }
    
    def _default_config(self) -> EnvironmentConfig:
        """Create default environment configuration."""
        return EnvironmentConfig(
            name="MetaOpenEnv",
            version="1.0.0",
            description="Real-world task simulation environment with email triage, code review, and content moderation",
            max_episode_steps=100,
            action_space_description="Actions include classify, flag, approve, request_changes, skip, delete",
            observation_space_description="Task-specific observations with data dictionaries and metadata",
            tasks=[
                {
                    "name": "Email Triage",
                    "type": "email_triage",
                    "difficulty": "easy",
                    "description": "Classify emails into categories (work, personal, spam, promotional)",
                },
                {
                    "name": "Code Review",
                    "type": "code_review",
                    "difficulty": "medium",
                    "description": "Review code for bugs and issues, decide on approval or changeset",
                },
                {
                    "name": "Content Moderation",
                    "type": "content_moderation",
                    "difficulty": "hard",
                    "description": "Moderate content for harmful or policy-violating posts",
                },
            ],
        )
    
    def reset(self, task_type: TaskType = None, task_id: str = None) -> Observation:
        """
        Reset the environment and start a new episode.
        
        Args:
            task_type: Type of task (EmailTriage, CodeReview, or ContentModeration)
            task_id: Optional custom task ID
            
        Returns:
            Initial observation
        """
        if task_type is None:
            task_type = TaskType.EMAIL_TRIAGE  # Default task
        
        if task_type not in self.task_registry:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Initialize task
        task_id = task_id or f"task_{datetime.now().isoformat()}"
        task_class = self.task_registry[task_type]
        
        # Determine difficulty
        difficulty_map = {
            TaskType.EMAIL_TRIAGE: "easy",
            TaskType.CODE_REVIEW: "medium",
            TaskType.CONTENT_MODERATION: "hard",
        }
        difficulty = difficulty_map.get(task_type, "medium")
        
        self.current_task = task_class(task_id, difficulty)
        self.current_task_type = task_type
        self.episode_step = 0
        self.episode_reward = 0.0
        self.trajectory = []
        self.reward_function.reset()
        
        # Get initial observation data
        observation_data = self.current_task.reset()
        
        return Observation(
            task_id=task_id,
            task_type=task_type,
            task_name=str(task_type.value).replace("_", " ").title(),
            data=observation_data,
            step_count=0,
            metadata={"difficulty": difficulty},
        )
    
    def step(self, action: Action) -> StepResult:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            StepResult with (observation, reward, done, info)
        """
        if self.current_task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.episode_step += 1
        
        # Convert action to dict and preserve all fields
        action_dict = action.model_dump(exclude_none=False)
        
        # Execute action in task
        reward_value, done, task_info, next_obs_data = self.current_task.step(action_dict)
        
        # Compute detailed reward
        reward_result = self.reward_function.compute_reward(
            action=action_dict,
            action_result={"action_reward": reward_value, "success": task_info.get("correct", False)},
            step_count=self.episode_step,
            done=done,
            episode_data={"max_steps": self.config.max_episode_steps},
        )
        
        self.episode_reward += reward_result["value"]
        
        # Build observation
        observation = Observation(
            task_id=self.current_task.task_id,
            task_type=self.current_task_type,
            task_name=str(self.current_task_type.value).replace("_", " ").title(),
            data=next_obs_data,
            step_count=self.episode_step,
            metadata=task_info,
        )
        
        # Build reward
        reward = Reward(
            value=reward_result["value"],
            success=task_info.get("correct", reward_result["value"] > 0),
            message=f"Action: {action.action_type} | Reward: {reward_result['value']:.3f}",
            components=reward_result["components"],
        )
        
        # Record trajectory
        self.trajectory.append({
            "step": self.episode_step,
            "observation": observation.model_dump(),
            "action": action_dict,
            "reward": reward_result["value"],
        })
        
        # Determine if done
        if done or self.episode_step >= self.config.max_episode_steps:
            done = True
        
        info = {
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "action_info": task_info,
            "reward_components": reward_result["components"],
            "trajectory_length": len(self.trajectory),
        }
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    
    def state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        
        Returns:
            Current state dictionary
        """
        if self.current_task is None:
            return {
                "status": "not_initialized",
                "message": "Call reset() to initialize environment",
            }
        
        return {
            "status": "active",
            "task_id": self.current_task.task_id,
            "task_type": str(self.current_task_type.value),
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "trajectory_length": len(self.trajectory),
            "task_state": self.current_task.state,
            "max_steps": self.config.max_episode_steps,
        }
    
    def evaluate_episode(self) -> Dict[str, Any]:
        """
        Evaluate the completed episode using the task grader.
        
        Returns:
            Evaluation results with final score
        """
        if self.current_task is None or not self.trajectory:
            return {"error": "No episode to evaluate"}
        
        grader = self.grader_registry.get(self.current_task_type)
        if not grader:
            return {"error": f"No grader for task type {self.current_task_type}"}
        
        final_state = self.current_task.state.copy()
        final_score = grader.grade(self.trajectory, final_state)
        
        stats = self.reward_function.get_trajectory_stats()
        
        return {
            "task_id": self.current_task.task_id,
            "task_type": str(self.current_task_type.value),
            "difficulty": self.current_task.difficulty,
            "final_score": final_score,  # 0.0 to 1.0
            "episode_steps": self.episode_step,
            "episode_reward": self.episode_reward,
            "trajectory_steps": stats["total_steps"],
            "average_reward_per_step": stats["average_reward_per_step"],
            "positive_steps": stats["positive_steps"],
            "negative_steps": stats["negative_steps"],
        }
    
    def render(self, mode: str = "dict") -> str:
        """
        Render the current state in human-readable format.
        
        Args:
            mode: Render mode ("dict", "json", or "text")
            
        Returns:
            Rendered state
        """
        state = self.state()
        
        if mode == "json":
            return json.dumps(state, indent=2, default=str)
        elif mode == "dict":
            return str(state)
        elif mode == "text":
            return self._text_render()
        else:
            return str(state)
    
    def _text_render(self) -> str:
        """Render in human-readable text format."""
        state = self.state()
        lines = [
            f"=== OpenEnv Environment ===",
            f"Task: {state.get('task_type', 'N/A')}",
            f"Episode Step: {state.get('episode_step', 0)}",
            f"Episode Reward: {state.get('episode_reward', 0.0):.3f}",
            f"Status: {state.get('status', 'unknown')}",
        ]
        return "\n".join(lines)
    
    def get_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config.model_dump()
