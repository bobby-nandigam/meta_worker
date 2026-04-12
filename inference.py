#!/usr/bin/env python3
"""
Inference Script for Meta OpenEnv - Follows exact hackathon specification
"""

import os
import json
import sys
from typing import Dict, List, Optional
from datetime import datetime

# Import with error handling
try:
    from openai import OpenAI
    from pydantic import BaseModel
    
    # Import environment and models
    from openenv import (
        OpenEnv, Observation, Action, Reward, TaskType
    )
except ImportError as e:
    # Import failed - print guaranteed output in EXACT spec format before exiting
    print("[START] task=import env=meta_openenv model=gpt-3.5-turbo", flush=True)
    print("[STEP] step=1 action=none reward=0.00 done=true error=null", flush=True)
    print("[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
    sys.exit(0)

# Environment variables - EXACT as per spec
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "meta_openenv"
MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 500


class BaselineConfig(BaseModel):
    """Configuration for baseline evaluation"""
    model_name: str = MODEL_NAME
    api_base_url: str = API_BASE_URL
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    num_episodes: int = 3
    timeout_seconds: int = 30


# LOGGING FUNCTIONS - EXACT FORMAT PER SPEC
def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in exact spec format"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log step in exact spec format"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end in exact spec format"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


class InferenceClient:
    """Client for calling LLM inference via OpenAI"""
    
    def __init__(self, api_key: str = None, api_base: str = None, model: str = None):
        """Initialize OpenAI client"""
        self.api_key = api_key or HF_TOKEN
        self.api_base = api_base or API_BASE_URL
        self.model = model or MODEL_NAME
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
    
    def call_model(self, prompt: str) -> str:
        """Call inference model via OpenAI client"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            raise


class AutonomousAgent:
    """
    Baseline agent that uses LLM reasoning to solve tasks
    Uses few-shot prompting to guide decision-making
    """
    
    def __init__(self, inference_client: InferenceClient):
        self.client = inference_client
        self.action_history = []
    
    def decide_action(self, observation: Observation) -> Action:
        """
        Decide next action based on observation
        Uses LLM reasoning for intelligent decision-making
        """
        prompt = self._build_decision_prompt(observation)
        
        # Call LLM for decision
        response = self.client.call_model(prompt)
        
        # Parse response into action
        action = self._parse_action_response(response, observation)
        
        self.action_history.append({
            "step": observation.step_count,
            "action": action.action_type,
            "confidence": action.confidence
        })
        
        return action
    
    def _build_decision_prompt(self, observation: Observation) -> str:
        """Build few-shot prompt for decision-making"""
        
        # task_type is already a string because Observation uses use_enum_values=True
        task_type = observation.task_type if isinstance(observation.task_type, str) else observation.task_type.value
        
        if task_type == "email_triage":
            return self._email_triage_prompt(observation)
        elif task_type == "code_review":
            return self._code_review_prompt(observation)
        elif task_type == "content_moderation":
            return self._content_moderation_prompt(observation)
        else:
            return ""
    
    def _email_triage_prompt(self, observation: Observation) -> str:
        """Few-shot prompt for email triage"""
        return f"""
You are an AI agent for email automation. Analyze the email below and decide the action.

Current Email:
{json.dumps(observation.data.get('current_email', {}), indent=2)}

## Few-Shot Examples:

Example 1:
Email: From: ceo@meta.com, Subject: "Urgent: Board meeting"
Decision: {{"action_type": "classify", "confidence": 0.95, "classification": "work"}}

Example 2:
Email: From: friend@personal.com, Subject: "Let's catch up"
Decision: {{"action_type": "classify", "confidence": 0.90, "classification": "personal"}}

Example 3:
Email: From: spam@vendor.com, Subject: "CLICK HERE: Amazing offer!!!"
Decision: {{"action_type": "classify", "confidence": 0.98, "classification": "spam"}}

Now analyze the current email and respond with JSON only:
{{
  "action_type": "classify",
  "classification": "work|personal|spam|promotional",
  "confidence": 0.85,
  "reasoning": "brief explanation"
}}
"""
    
    def _code_review_prompt(self, observation: Observation) -> str:
        """Few-shot prompt for code review"""
        return f"""
You are an AI code reviewer. Analyze the code change and identify issues.

Code Diff:
{json.dumps(observation.data.get('code_snippet', ''), indent=2)}

## Few-Shot Examples:

Example 1: Missing type hints
Decision: {{"action_type": "flag", "description": "Missing type hints", "confidence": 0.9}}

Example 2: Security vulnerability
Decision: {{"action_type": "flag", "description": "Hardcoded secret", "confidence": 0.95}}

Now analyze and respond with JSON:
{{
  "action_type": "flag|approve",
  "description": "finding",
  "confidence": 0.85
}}
"""
    
    def _content_moderation_prompt(self, observation: Observation) -> str:
        """Few-shot prompt for content moderation"""
        return f"""
You are a content moderator. Analyze the content and decide on moderation action.

Content:
{json.dumps(observation.data.get('content', {}), indent=2)}

## Few-Shot Examples:

Example 1: Harmful content
Decision: {{"action_type": "flag", "classification": "harmful", "confidence": 0.95}}

Example 2: Safe content
Decision: {{"action_type": "approve", "classification": "safe", "confidence": 0.90}}

Now analyze and respond with JSON:
{{
  "action_type": "approve|flag",
  "classification": "safe|harmful|misleading",
  "confidence": 0.85
}}
"""
    
    def _parse_action_response(self, response: str, observation: Observation) -> Action:
        """Parse LLM response into Action object"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                action_json = response[json_start:json_end]
                action_data = json.loads(action_json)
            else:
                action_data = {}
            
            # Create Action with available fields
            action = Action(
                action_type=action_data.get('action_type', 'approve'),
                confidence=float(action_data.get('confidence', 0.5)),
                reasoning=action_data.get('reasoning', ''),
                classification=action_data.get('classification', ''),
            )
            return action
            
        except Exception as e:
            # Fallback to default action
            return Action(
                action_type="approve",
                confidence=0.3,
                reasoning="Fallback action due to parsing error",
            )


def evaluate_task(
    task_type: TaskType, 
    inference_client: InferenceClient,
    num_episodes: int = 3
) -> Dict[str, float]:
    """Evaluate agent performance on a specific task"""
    
    log_start(task=task_type.value, env=BENCHMARK, model=MODEL_NAME or "gpt-3.5-turbo")
    
    agent = AutonomousAgent(inference_client)
    episode_scores = []
    all_rewards = []
    total_steps = 0
    success = False
    
    try:
        for episode in range(num_episodes):
            try:
                # Initialize environment (no task_type here - goes in reset)
                env = OpenEnv()
                
                # task_type is already a TaskType enum
                observation = env.reset(task_type=task_type)
                
                done = False
                step_count = 0
                
                while not done and step_count < MAX_STEPS:
                    try:
                        # Agent decides action
                        action = agent.decide_action(observation)
                        
                        # Execute action
                        observation, reward, done, info = env.step(action)
                        
                        step_count += 1
                        total_steps += 1
                        
                        # Extract reward value
                        episode_reward = reward.immediate_reward if hasattr(reward, 'immediate_reward') else 0.0
                        all_rewards.append(episode_reward)
                        
                        # Log step in EXACT spec format
                        action_str = str(action.action_type) if hasattr(action, 'action_type') else "action"
                        error_msg = None
                        log_step(
                            step=total_steps,
                            action=action_str,
                            reward=episode_reward,
                            done=done,
                            error=error_msg
                        )
                    except Exception as step_error:
                        # Log step error but continue
                        total_steps += 1
                        log_step(
                            step=total_steps,
                            action="error",
                            reward=0.0,
                            done=True,
                            error=str(step_error)[:50]
                        )
                        break
                
                # Ensure at least 1 step was logged
                if total_steps == 0:
                    total_steps = 1
                    log_step(step=1, action="init", reward=0.0, done=True, error="no steps executed")
                
                # Grade episode
                try:
                    score = env.grade()
                    episode_scores.append(score)
                except:
                    episode_scores.append(0.0)
            except Exception as ep_error:
                # Episode failed, log it and continue
                if total_steps == 0:
                    total_steps = 1
                    log_step(step=1, action="init", reward=0.0, done=True, error=str(ep_error)[:50])
                pass
        
        # Compute statistics
        avg_score = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
        avg_score = min(max(avg_score, 0.0), 1.0)  # Clamp to [0, 1]
        success = avg_score > 0.5
        
    except Exception as e:
        # On exception, use defaults
        avg_score = 0.0
        success = False
        total_steps = max(1, total_steps)
        all_rewards = all_rewards if all_rewards else [0.0]
    
    finally:
        # ALWAYS emit [END], even on exception (per spec)
        log_end(
            success=success,
            steps=total_steps,
            score=avg_score,
            rewards=all_rewards
        )
    
    return {
        "task_type": task_type,
        "num_episodes": num_episodes,
        "scores": episode_scores if 'episode_scores' in locals() else [],
        "average": avg_score if 'avg_score' in locals() else 0.0,
    }


def main():
    """Main evaluation loop"""
    
    config = BaselineConfig()
    
    # Check if HF_TOKEN is available
    if not HF_TOKEN:
        log_start(task="error", env=BENCHMARK, model=MODEL_NAME or "gpt-3.5-turbo")
        log_step(step=1, action="none", reward=0.0, done=True, error="HF_TOKEN not set")
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return
    
    # Initialize inference client
    try:
        client = InferenceClient()
    except Exception as e:
        log_start(task="error", env=BENCHMARK, model=MODEL_NAME or "gpt-3.5-turbo")
        log_step(step=1, action="none", reward=0.0, done=True, error=str(e))
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return
    
    # Evaluate all tasks
    task_types = [TaskType.EMAIL_TRIAGE, TaskType.CODE_REVIEW, TaskType.CONTENT_MODERATION]
    
    for task_type in task_types:
        try:
            evaluate_task(
                task_type=task_type,
                inference_client=client,
                num_episodes=config.num_episodes
            )
        except Exception as e:
            # log_end is called in finally block inside evaluate_task
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Last resort fallback - ensure output in spec format
        log_start(task="error", env=BENCHMARK, model=MODEL_NAME or "gpt-3.5-turbo")
        log_step(step=1, action="none", reward=0.0, done=True, error=str(e))
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
