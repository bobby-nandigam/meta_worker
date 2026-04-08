#!/usr/bin/env python3
"""
Test suite for MetaOpenEnv validation.

Validates OpenEnv specification compliance and environment functionality.
"""

import sys
import traceback
from typing import List, Tuple

from openenv import (
    OpenEnv, TaskType, Action, Observation, Reward, StepResult,
    EmailTriageTask, CodeReviewTask, ContentModerationTask,
    EmailTriageGrader, CodeReviewGrader, ContentModerationGrader,
)


class TestRunner:
    """Run tests and report results."""
    
    def __init__(self):
        self.tests: List[Tuple[str, callable]] = []
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_test(self, name: str, test_func: callable):
        """Register a test."""
        self.tests.append((name, test_func))
    
    def run_all(self) -> bool:
        """Run all tests and return success status."""
        print("=" * 70)
        print("MetaOpenEnv Specification Validation")
        print("=" * 70)
        
        for name, test_func in self.tests:
            try:
                test_func()
                print(f"✓ {name}")
                self.passed += 1
            except AssertionError as e:
                print(f"✗ {name}")
                print(f"  Error: {e}")
                self.failed += 1
                self.errors.append((name, str(e)))
            except Exception as e:
                print(f"✗ {name} (Exception)")
                print(f"  Error: {type(e).__name__}: {e}")
                self.failed += 1
                self.errors.append((name, traceback.format_exc()))
        
        print("\n" + "=" * 70)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        
        return self.failed == 0


def test_openenv_initialization():
    """Test OpenEnv can be initialized."""
    env = OpenEnv()
    assert env is not None, "Failed to initialize OpenEnv"
    assert env.config is not None, "Config not initialized"


def test_reset_returns_observation():
    """Test reset() returns Observation."""
    env = OpenEnv()
    obs = env.reset(task_type=TaskType.EMAIL_TRIAGE)
    assert isinstance(obs, Observation), f"Expected Observation, got {type(obs)}"
    assert obs.task_id is not None, "task_id is None"
    assert obs.task_type == TaskType.EMAIL_TRIAGE, "task_type mismatch"
    assert obs.data is not None, "data is None"


def test_reset_all_task_types():
    """Test reset with all task types."""
    env = OpenEnv()
    for task_type in [TaskType.EMAIL_TRIAGE, TaskType.CODE_REVIEW, TaskType.CONTENT_MODERATION]:
        obs = env.reset(task_type=task_type)
        assert obs.task_type == task_type, f"Expected {task_type}, got {obs.task_type}"


def test_pydantic_models():
    """Test Pydantic models."""
    # Test Observation
    obs = Observation(
        task_id="test",
        task_type=TaskType.EMAIL_TRIAGE,
        task_name="Test",
        data={},
        step_count=0
    )
    assert hasattr(obs, 'model_dump'), "Observation missing model_dump"
    obs_dict = obs.model_dump()
    assert isinstance(obs_dict, dict), "model_dump should return dict"
    
    # Test Action
    action = Action(action_type="classify", target_id="test")
    assert hasattr(action, 'model_dump'), "Action missing model_dump"
    
    # Test Reward
    reward = Reward(value=0.5, success=True, message="test")
    assert hasattr(reward, 'model_dump'), "Reward missing model_dump"


def test_step_returns_step_result():
    """Test step() returns StepResult."""
    env = OpenEnv()
    env.reset(task_type=TaskType.EMAIL_TRIAGE)
    
    action = Action(
        action_type="classify",
        target_id="email_0",
        classification="work"
    )
    
    result = env.step(action)
    assert isinstance(result, StepResult), f"Expected StepResult, got {type(result)}"
    assert isinstance(result.observation, Observation), "observation not Observation"
    assert isinstance(result.reward, Reward), "reward not Reward"
    assert isinstance(result.done, bool), "done not bool"
    assert isinstance(result.info, dict), "info not dict"


def test_state_returns_dict():
    """Test state() returns dict."""
    env = OpenEnv()
    env.reset(task_type=TaskType.EMAIL_TRIAGE)
    
    state = env.state()
    assert isinstance(state, dict), f"Expected dict, got {type(state)}"
    assert "status" in state, "status not in state"
    assert "task_id" in state, "task_id not in state"


def test_action_validation():
    """Test Action model validation."""
    # Valid action
    action = Action(
        action_type="classify",
        target_id="email_0",
        classification="work",
        confidence=0.95
    )
    assert action.confidence == 0.95, "confidence not set"
    
    # Confidence bounds
    try:
        bad_action = Action(
            action_type="classify",
            confidence=1.5  # Invalid: > 1.0
        )
        assert False, "Should reject confidence > 1.0"
    except (ValueError, Exception):
        pass  # Expected


def test_reward_validity():
    """Test Reward model."""
    reward = Reward(
        value=0.5,
        success=True,
        message="Good job",
        components={"action_reward": 0.3, "bonus": 0.2}
    )
    assert reward.value == 0.5, "value mismatch"
    assert reward.components["action_reward"] == 0.3, "components mismatch"


def test_email_triage_task():
    """Test Email Triage task."""
    env = OpenEnv()
    obs = env.reset(task_type=TaskType.EMAIL_TRIAGE)
    
    # Check observation structure
    assert "emails" in obs.data, "emails not in observation"
    assert "classified_count" in obs.data, "classified_count not in observation"
    assert "total_emails" in obs.data, "total_emails not in observation"
    
    # Take some actions
    action = Action(
        action_type="classify",
        target_id="email_0",
        classification="work"
    )
    result = env.step(action)
    assert not result.done, "Episode should not be done after 1 step"
    assert result.reward.value > 0, "Should get positive reward for correct action"


def test_code_review_task():
    """Test Code Review task."""
    env = OpenEnv()
    obs = env.reset(task_type=TaskType.CODE_REVIEW)
    
    # Check observation structure
    assert "code_blocks" in obs.data, "code_blocks not in observation"
    assert "flagged_count" in obs.data, "flagged_count not in observation"
    
    # Flag an issue
    action = Action(action_type="flag_issue", target_id="issue_0")
    result = env.step(action)
    assert isinstance(result.reward.value, float), "reward value not float"


def test_content_moderation_task():
    """Test Content Moderation task."""
    env = OpenEnv()
    obs = env.reset(task_type=TaskType.CONTENT_MODERATION)
    
    # Check observation structure
    assert "items" in obs.data, "items not in observation"
    assert "reviewed_count" in obs.data, "reviewed_count not in observation"
    
    # Flag content
    action = Action(action_type="flag_content", target_id="item_0")
    result = env.step(action)
    assert isinstance(result.reward, Reward), "reward not Reward"


def test_episode_completion():
    """Test completing a full episode."""
    env = OpenEnv()
    obs = env.reset(task_type=TaskType.EMAIL_TRIAGE)
    
    done = False
    step_count = 0
    max_steps = 50
    
    while not done and step_count < max_steps:
        remaining = obs.data.get("remaining", [])
        if not remaining:
            # All emails classified, force completion
            break
        
        email = remaining[0]
        action = Action(
            action_type="classify",
            target_id=email["id"],
            classification="work"
        )
        
        result = env.step(action)
        obs = result.observation
        done = result.done
        step_count += 1
    
    # Must provide evaluation at end
    evaluation = env.evaluate_episode()
    assert "final_score" in evaluation, "evaluation missing final_score"
    assert 0.0 <= evaluation["final_score"] <= 1.0, f"Invalid score: {evaluation['final_score']}"


def test_graders_produce_valid_scores():
    """Test graders produce scores in [0.0, 1.0]."""
    env = OpenEnv()
    
    for task_type in [TaskType.EMAIL_TRIAGE, TaskType.CODE_REVIEW, TaskType.CONTENT_MODERATION]:
        obs = env.reset(task_type=task_type)
        
        # Take some random valid actions
        for _ in range(5):
            if task_type == TaskType.EMAIL_TRIAGE:
                action = Action(action_type="classify", target_id="email_0", classification="work")
            elif task_type == TaskType.CODE_REVIEW:
                action = Action(action_type="flag_issue", target_id="issue_0")
            else:  # Content Moderation
                action = Action(action_type="flag_content", target_id="item_0")
            
            result = env.step(action)
            if result.done:
                break
        
        evaluation = env.evaluate_episode()
        score = evaluation.get("final_score")
        assert score is not None, f"Grader returned None for {task_type}"
        assert isinstance(score, float), f"Score not float: {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_reward_trajectory_tracking():
    """Test reward function tracks trajectory."""
    env = OpenEnv()
    obs = env.reset(task_type=TaskType.EMAIL_TRIAGE)
    
    total_reward = 0.0
    for _ in range(5):
        action = Action(action_type="skip_review" if _ % 2 == 0 else "classify",
                       target_id="email_0", classification="work")
        result = env.step(action)
        total_reward += result.reward.value
    
    state = env.state()
    assert state["episode_reward"] > -10, "Reward tracking error"


def test_environment_config():
    """Test environment configuration."""
    env = OpenEnv()
    config = env.get_config()
    
    assert "name" in config, "name missing from config"
    assert "version" in config, "version missing from config"
    assert "tasks" in config, "tasks missing from config"
    assert config["name"] == "MetaOpenEnv", "wrong environment name"
    assert len(config["tasks"]) >= 3, "should have at least 3 tasks"


def test_observation_step_count():
    """Test observation step_count increments."""
    env = OpenEnv()
    obs1 = env.reset(task_type=TaskType.EMAIL_TRIAGE)
    assert obs1.step_count == 0, "initial step_count not 0"
    
    action = Action(action_type="skip_review", target_id="email_0")
    result = env.step(action)
    obs2 = result.observation
    assert obs2.step_count == 1, "step_count not incremented"


def test_max_episode_steps_enforcement():
    """Test max_episode_steps is enforced."""
    env = OpenEnv()
    obs = env.reset(task_type=TaskType.EMAIL_TRIAGE)
    
    max_steps = env.config.max_episode_steps
    done = False
    
    for i in range(max_steps + 10):
        action = Action(action_type="skip_review", target_id="email_0")
        result = env.step(action)
        if result.done or i >= max_steps - 1:
            done = True
            break
    
    assert done, "Episode should end after max_steps"


def main():
    """Run all tests."""
    runner = TestRunner()
    
    # Add all tests
    tests = [
        ("OpenEnv initialization", test_openenv_initialization),
        ("reset() returns Observation", test_reset_returns_observation),
        ("reset() with all task types", test_reset_all_task_types),
        ("Pydantic models (Observation, Action, Reward)", test_pydantic_models),
        ("step() returns StepResult", test_step_returns_step_result),
        ("state() returns dict", test_state_returns_dict),
        ("Action model validation", test_action_validation),
        ("Reward model validity", test_reward_validity),
        ("Email Triage task", test_email_triage_task),
        ("Code Review task", test_code_review_task),
        ("Content Moderation task", test_content_moderation_task),
        ("Episode completion", test_episode_completion),
        ("Graders produce valid scores", test_graders_produce_valid_scores),
        ("Reward trajectory tracking", test_reward_trajectory_tracking),
        ("Environment configuration", test_environment_config),
        ("Observation step_count", test_observation_step_count),
        ("Max episode steps enforcement", test_max_episode_steps_enforcement),
    ]
    
    for name, test_func in tests:
        runner.add_test(name, test_func)
    
    success = runner.run_all()
    
    # Print errors if any
    if runner.errors:
        print("\nDetailed Errors:")
        for name, error in runner.errors:
            print(f"\n{name}:")
            print(f"  {error}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
