"""Reward function implementation for the OpenEnv environment."""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    
    # Positive rewards
    correct_action_reward: float = 0.1
    progress_reward: float = 0.05
    efficiency_bonus: float = 0.2
    completion_bonus: float = 0.5
    
    # Negative rewards
    incorrect_action_penalty: float = -0.05
    destructive_action_penalty: float = -0.5
    inefficiency_penalty: float = -0.1
    infinite_loop_penalty: float = -1.0


class RewardFunction:
    """Implements the reward function for the environment."""
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.action_history = []
        self.step_rewards = []
    
    def compute_reward(
        self,
        action: Dict[str, Any],
        action_result: Dict[str, Any],
        step_count: int,
        done: bool,
        episode_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compute reward for an action.
        
        Args:
            action: The action taken
            action_result: Result of the action (success, info, etc.)
            step_count: Current step in episode
            done: Whether episode is done
            episode_data: Additional episode metadata
            
        Returns:
            Dict with 'value' and 'components' breakdown
        """
        reward_components = {}
        
        # Base reward from action result
        action_reward = action_result.get("action_reward", 0.0)
        reward_components["action_reward"] = action_reward
        
        # Penalize destructive actions
        if action.get("action_type") in ["delete", "ban", "permanent_delete"]:
            reward_components["destructive_penalty"] = self.config.destructive_action_penalty
        else:
            reward_components["destructive_penalty"] = 0.0
        
        # Reward progress (non-zero reward steps)
        if action_reward > 0:
            reward_components["progress_reward"] = self.config.progress_reward
        else:
            reward_components["progress_reward"] = 0.0
        
        # Penalize infinite loops (repeated actions)
        if self._is_suspicious_loop(action):
            reward_components["loop_penalty"] = self.config.infinite_loop_penalty * 0.1
        else:
            reward_components["loop_penalty"] = 0.0
        
        # Efficiency bonus at episode end
        if done:
            efficiency_bonus = self._compute_efficiency_bonus(step_count, episode_data)
            reward_components["efficiency_bonus"] = efficiency_bonus
            
            # Completion bonus
            if action_result.get("success", False):
                reward_components["completion_bonus"] = self.config.completion_bonus
            else:
                reward_components["completion_bonus"] = 0.0
        else:
            reward_components["efficiency_bonus"] = 0.0
            reward_components["completion_bonus"] = 0.0
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        # Store for trajectory tracking
        self.step_rewards.append({
            "step": step_count,
            "value": total_reward,
            "components": reward_components.copy(),
        })
        
        return {
            "value": total_reward,
            "components": reward_components,
            "cumulative": sum(r["value"] for r in self.step_rewards),
        }
    
    def _is_suspicious_loop(self, current_action: Dict[str, Any]) -> bool:
        """Detect potential infinite loops by checking repeated actions."""
        if not self.action_history:
            self.action_history.append(current_action)
            return False
        
        # Check if same action repeated 5+ times in a row
        if len(self.action_history) >= 5:
            last_five = self.action_history[-5:]
            action_type = current_action.get("action_type")
            if all(a.get("action_type") == action_type for a in last_five):
                return True
        
        self.action_history.append(current_action)
        return False
    
    def _compute_efficiency_bonus(self, step_count: int, episode_data: Dict[str, Any]) -> float:
        """Compute efficiency bonus based on steps taken."""
        max_steps = episode_data.get("max_steps", 100)
        optimal_steps = episode_data.get("optimal_steps", max_steps // 2)
        
        if step_count <= optimal_steps:
            # Full efficiency bonus
            return self.config.efficiency_bonus
        elif step_count <= max_steps:
            # Partial efficiency bonus
            ratio = (max_steps - step_count) / (max_steps - optimal_steps)
            return self.config.efficiency_bonus * max(0.0, ratio)
        else:
            # Penalty for exceeding max steps (shouldn't happen but be safe)
            return -self.config.inefficiency_penalty
    
    def reset(self):
        """Reset reward tracking for new episode."""
        self.action_history = []
        self.step_rewards = []
    
    def get_trajectory_stats(self) -> Dict[str, Any]:
        """Get statistics about the trajectory."""
        if not self.step_rewards:
            return {
                "total_steps": 0,
                "total_reward": 0.0,
                "average_reward_per_step": 0.0,
                "positive_steps": 0,
                "negative_steps": 0,
            }
        
        total_reward = sum(r["value"] for r in self.step_rewards)
        positive_steps = sum(1 for r in self.step_rewards if r["value"] > 0)
        negative_steps = sum(1 for r in self.step_rewards if r["value"] < 0)
        
        return {
            "total_steps": len(self.step_rewards),
            "total_reward": total_reward,
            "average_reward_per_step": total_reward / len(self.step_rewards),
            "positive_steps": positive_steps,
            "negative_steps": negative_steps,
            "reward_timeline": [r["value"] for r in self.step_rewards],
        }
