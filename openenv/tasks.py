"""Task definitions for the OpenEnv environment."""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import random
import json


class Task(ABC):
    """Base class for all tasks."""
    
    def __init__(self, task_id: str, difficulty: str):
        self.task_id = task_id
        self.difficulty = difficulty  # easy, medium, hard
        self.state = {}
        self.initial_state = {}
        self.step_count = 0
        self.max_steps = 100
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Initialize the task and return initial observation data."""
        self.step_count = 0
        return self.state.copy()
    
    @abstractmethod
    def step(self, action: Dict[str, Any]) -> tuple:
        """
        Execute an action and return (reward, done, info, next_state).
        """
        self.step_count += 1
        pass
    
    @abstractmethod
    def get_observation_data(self) -> Dict[str, Any]:
        """Return current observation data."""
        pass


class EmailTriageTask(Task):
    """Easy task: Email triage and classification."""
    
    EMAIL_CATEGORIES = ["work", "personal", "spam", "promotional"]
    
    def __init__(self, task_id: str, difficulty: str = "easy"):
        super().__init__(task_id, difficulty)
        self.emails = []
        self.correct_labels = {}
        self.classifications = {}
    
    def reset(self) -> Dict[str, Any]:
        """Generate initial emails to classify."""
        super().reset()
        
        num_emails = 8
        self.emails = [
            {
                "id": f"email_{i}",
                "subject": self._generate_subject(i),
                "preview": self._generate_preview(i),
                "sender": self._generate_sender(i),
            }
            for i in range(num_emails)
        ]
        
        # Assign correct labels
        self.correct_labels = {
            "email_0": "work",
            "email_1": "spam",
            "email_2": "personal",
            "email_3": "promotional",
            "email_4": "work",
            "email_5": "spam",
            "email_6": "personal",
            "email_7": "promotional",
        }
        
        self.classifications = {}
        self.state = {
            "emails": self.emails,
            "classifications": self.classifications,
            "correct_labels": self.correct_labels,
            "total_emails": num_emails,
        }
        return self.get_observation_data()
    
    def step(self, action: Dict[str, Any]) -> tuple:
        """Process classification action."""
        super().step(action)
        
        action_type = action.get("action_type")
        target_id = action.get("target_id")
        
        reward_value = 0.0
        info = {"action_type": action_type}
        done = False
        success = False
        
        if action_type == "classify":
            classification = action.get("classification")
            if target_id in self.correct_labels:
                if classification == self.correct_labels[target_id]:
                    reward_value = 0.1
                    success = True
                else:
                    reward_value = -0.05
                
                self.classifications[target_id] = classification
                info["correct"] = success
        
        elif action_type == "skip":
            reward_value = 0.02
            success = True
            info["reason"] = "skipped"
        
        elif action_type == "delete":
            reward_value = -0.5  # Penalize destructive actions
            info["reason"] = "destructive_action"
        
        # Check if done
        if len(self.classifications) == len(self.emails):
            done = True
            # Bonus for completing efficiently
            if self.step_count <= 8:
                reward_value += 0.2
        
        if self.step_count >= self.max_steps:
            done = True
        
        return reward_value, done, info, self.get_observation_data()
    
    def get_observation_data(self) -> Dict[str, Any]:
        """Return current observation."""
        return {
            "emails": self.emails,
            "classified_count": len(self.classifications),
            "total_emails": len(self.emails),
            "remaining": [e for e in self.emails if e["id"] not in self.classifications],
        }
    
    def _generate_subject(self, idx: int) -> str:
        subjects = [
            "Quarterly Review Meeting",
            "URGENT: Get rich quick!",
            "Coffee plans this weekend?",
            "50% OFF Everything Today!",
            "Budget Updates for Q2",
            "Congratulations! You've won!",
            "Family Dinner Photos",
            "Limited Time Offer - Act Now!",
        ]
        return subjects[idx % len(subjects)]
    
    def _generate_preview(self, idx: int) -> str:
        previews = [
            "Please see the attached quarterly review...",
            "Click here to claim your prize now!",
            "Let's plan a meetup for this Saturday...",
            "Don't miss our exclusive sale...",
            "We need to discuss the budget allocation...",
            "You have been selected...",
            "Here are the photos from last week...",
            "This offer expires today!",
        ]
        return previews[idx % len(previews)]
    
    def _generate_sender(self, idx: int) -> str:
        senders = [
            "boss@company.com",
            "noreply@spam.com",
            "alice@personal.com",
            "promo@store.com",
            "team@company.com",
            "winner@scam.com",
            "mom@family.com",
            "deals@marketing.com",
        ]
        return senders[idx % len(senders)]


class CodeReviewTask(Task):
    """Medium task: Code review with issue identification."""
    
    def __init__(self, task_id: str, difficulty: str = "medium"):
        super().__init__(task_id, difficulty)
        self.code_blocks = []
        self.issues = {}
        self.flagged_issues = set()
    
    def reset(self) -> Dict[str, Any]:
        """Generate code with bugs for review."""
        super().reset()
        
        self.code_blocks = [
            {
                "id": "block_0",
                "language": "python",
                "code": "def divide(a, b):\n    return a / b  # No zero check!",
            },
            {
                "id": "block_1",
                "language": "javascript",
                "code": "var x = 10;\nvar x = 20;  // Redeclaration",
            },
            {
                "id": "block_2",
                "language": "python",
                "code": "import time\nprint('hello')",
            },
            {
                "id": "block_3",
                "language": "python",
                "code": "data = [1,2,3]\nresult = data[10]  # Index out of bounds!",
            },
        ]
        
        self.issues = {
            "issue_0": {"id": "issue_0", "block_id": "block_0", "type": "logic", "is_bug": True, "severity": "high"},
            "issue_1": {"id": "issue_1", "block_id": "block_1", "type": "syntax", "is_bug": True, "severity": "medium"},
            "issue_2": {"id": "issue_2", "block_id": "block_2", "type": "style", "is_bug": False, "severity": "low"},
            "issue_3": {"id": "issue_3", "block_id": "block_3", "type": "logic", "is_bug": True, "severity": "high"},
        }
        
        self.flagged_issues = set()
        self.correct_decision = "request_changes"  # PR has bugs, should request changes
        
        self.state = {
            "code_blocks": self.code_blocks,
            "issues": self.issues,
            "total_bugs": sum(1 for i in self.issues.values() if i["is_bug"]),
        }
        
        return self.get_observation_data()
    
    def step(self, action: Dict[str, Any]) -> tuple:
        """Process code review action."""
        super().step(action)
        
        action_type = action.get("action_type")
        reward_value = 0.0
        info = {"action_type": action_type}
        done = False
        success = False
        
        if action_type == "flag_issue":
            issue_id = action.get("target_id")
            if issue_id in self.issues:
                issue = self.issues[issue_id]
                if issue["is_bug"]:
                    # Correct bug identification
                    reward_value = 0.15
                    success = True
                else:
                    # False positive on style issue
                    reward_value = -0.1
                
                self.flagged_issues.add(issue_id)
                info["is_bug"] = issue["is_bug"]
        
        elif action_type == "approve":
            # Check if decision is correct
            total_bugs = sum(1 for i in self.issues.values() if i["is_bug"])
            found_bugs = sum(1 for iid in self.flagged_issues if self.issues[iid]["is_bug"])
            
            if found_bugs == 0:
                reward_value = 0.2
                success = True
            else:
                reward_value = -0.3
            
            done = True
        
        elif action_type == "request_changes":
            # Check if decision is correct
            total_bugs = sum(1 for i in self.issues.values() if i["is_bug"])
            found_bugs = sum(1 for iid in self.flagged_issues if self.issues[iid]["is_bug"])
            
            if found_bugs > 0:
                reward_value = 0.25
                success = True
            else:
                reward_value = -0.2
            
            done = True
        
        if self.step_count >= self.max_steps:
            done = True
        
        return reward_value, done, info, self.get_observation_data()
    
    def get_observation_data(self) -> Dict[str, Any]:
        """Return current observation."""
        return {
            "code_blocks": self.code_blocks,
            "flagged_count": len(self.flagged_issues),
            "total_issues": len(self.issues),
        }


class ContentModerationTask(Task):
    """Hard task: Content moderation with nuanced policy enforcement."""
    
    def __init__(self, task_id: str, difficulty: str = "hard"):
        super().__init__(task_id, difficulty)
        self.items = {}
        self.flagged_items = set()
        self.max_steps = 150
    
    def reset(self) -> Dict[str, Any]:
        """Generate content for moderation."""
        super().reset()
        
        self.items = {
            "item_0": {
                "id": "item_0",
                "content": "Kill all [group]",
                "is_harmful": True,
                "category": "hate_speech",
                "severity": "critical",
            },
            "item_1": {
                "id": "item_1",
                "content": "I hate Mondays.",
                "is_harmful": False,
                "category": "hyperbole",
                "severity": "low",
            },
            "item_2": {
                "id": "item_2",
                "content": "Instructions for explosives...",
                "is_harmful": True,
                "category": "dangerous_content",
                "severity": "critical",
            },
            "item_3": {
                "id": "item_3",
                "content": "Check out my new blog!",
                "is_harmful": False,
                "category": "self_promotion",
                "severity": "low",
            },
            "item_4": {
                "id": "item_4",
                "content": "You're stupid and should die.",
                "is_harmful": True,
                "category": "harassment",
                "severity": "high",
            },
            "item_5": {
                "id": "item_5",
                "content": "Beautiful sunset today!",
                "is_harmful": False,
                "category": "neutral",
                "severity": "none",
            },
            "item_6": {
                "id": "item_6",
                "content": "Dr. Smith's research shows...",
                "is_harmful": False,
                "category": "informational",
                "severity": "none",
            },
            "item_7": {
                "id": "item_7",
                "content": "Follow my OnlyFans for exclusive content...",
                "is_harmful": True,
                "category": "sexual_content",
                "severity": "high",
            },
            "item_8": {
                "id": "item_8",
                "content": "Let's meet up for coffee!",
                "is_harmful": False,
                "category": "social_planning",
                "severity": "none",
            },
            "item_9": {
                "id": "item_9",
                "content": "[SCAM] Earn $10k per week...",
                "is_harmful": True,
                "category": "spam",
                "severity": "medium",
            },
        }
        
        self.flagged_items = set()
        self.state = {
            "items": self.items,
            "total_harmful_items": sum(1 for i in self.items.values() if i["is_harmful"]),
            "total_safe_items": sum(1 for i in self.items.values() if not i["is_harmful"]),
        }
        
        return self.get_observation_data()
    
    def step(self, action: Dict[str, Any]) -> tuple:
        """Process moderation action."""
        super().step(action)
        
        action_type = action.get("action_type")
        reward_value = 0.0
        info = {"action_type": action_type}
        done = False
        success = False
        
        if action_type == "flag_content":
            item_id = action.get("target_id")
            if item_id in self.items:
                item = self.items[item_id]
                if item["is_harmful"]:
                    # Correct detection
                    reward_value = 0.07
                    success = True
                else:
                    # False positive
                    reward_value = -0.08
                
                self.flagged_items.add(item_id)
                info["is_harmful"] = item["is_harmful"]
        
        elif action_type == "skip_review":
            reward_value = 0.01
            success = True
        
        # Check if done (reviewed all items)
        if len(self.flagged_items) == len(self.items):
            done = True
            # Calculate final bonus
            correct_flags = sum(1 for iid in self.flagged_items if self.items[iid]["is_harmful"])
            accuracy = correct_flags / len(self.flagged_items)
            if accuracy >= 0.8:
                reward_value += 0.3
        
        if self.step_count >= self.max_steps:
            done = True
        
        return reward_value, done, info, self.get_observation_data()
    
    def get_observation_data(self) -> Dict[str, Any]:
        """Return current observation."""
        # Hide the is_harmful flag in observation
        visible_items = {
            iid: {k: v for k, v in item.items() if k != "is_harmful"}
            for iid, item in self.items.items()
        }
        
        return {
            "items": visible_items,
            "reviewed_count": len(self.flagged_items),
            "total_items": len(self.items),
            "remaining": [iid for iid in self.items if iid not in self.flagged_items],
        }
