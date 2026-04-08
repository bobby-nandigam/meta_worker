"""Graders for evaluating agent performance on tasks."""

from typing import Dict, Any, List
from abc import ABC, abstractmethod
import json


class TaskGrader(ABC):
    """Base class for task-specific graders."""
    
    @abstractmethod
    def grade(self, trajectory: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
        """
        Grade the trajectory and final state.
        
        Args:
            trajectory: List of (observation, action, reward) tuples
            final_state: Final state after episode completion
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass


class EmailTriageGrader(TaskGrader):
    """Grader for email triage task."""
    
    def grade(self, trajectory: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
        """
        Evaluate email triage accuracy.
        
        Scoring:
        - Correct classifications: +0.1 per email (max 0.8 for 8 emails)
        - Efficiency: +0.2 bonus if done in ≤8 steps
        - No destructive actions: kept if no permanent deletions
        """
        correct_classifications = 0
        total_emails = final_state.get("total_emails", 8)
        actions_taken = len(trajectory)
        has_destructive_actions = any(
            action.get("action_type") == "delete" 
            for step in trajectory 
            for action in [step.get("action", {})]
        )
        
        # Count correct classifications
        for step in trajectory:
            action = step.get("action", {})
            if action.get("action_type") == "classify":
                prediction = action.get("classification")
                target_id = action.get("target_id")
                correct_label = final_state.get("correct_labels", {}).get(target_id)
                if prediction == correct_label:
                    correct_classifications += 1
        
        # Base score: accuracy
        accuracy_score = (correct_classifications / total_emails) * 0.8 if total_emails > 0 else 0.0
        
        # Efficiency bonus
        efficiency_bonus = 0.2 if actions_taken <= 8 else 0.0
        
        # Penalty for destructive actions
        destructive_penalty = 0.2 if has_destructive_actions else 0.0
        
        final_score = accuracy_score + efficiency_bonus - destructive_penalty
        return max(0.0, min(1.0, final_score))


class CodeReviewGrader(TaskGrader):
    """Grader for code review task."""
    
    def grade(self, trajectory: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
        """
        Evaluate code review quality.
        
        Scoring:
        - Correct bug identification: +0.15 per bug found (3 bugs, max 0.45)
        - Appropriate feedback: +0.3 if feedback is constructive and technical
        - Decision correctness: +0.25 if approve/request decision is correct
        """
        bugs_found = 0
        total_bugs = final_state.get("total_bugs", 3)
        
        # Track bugs found
        for step in trajectory:
            action = step.get("action", {})
            if action.get("action_type") == "flag_issue":
                issue_id = action.get("target_id")
                if final_state.get("issues", {}).get(issue_id, {}).get("is_bug"):
                    bugs_found += 1
        
        # Check if decision is correct
        final_decision = None
        for step in reversed(trajectory):
            action = step.get("action", {})
            if action.get("action_type") in ["approve", "request_changes"]:
                final_decision = action.get("action_type")
                break
        
        correct_decision = final_decision == final_state.get("correct_decision")
        
        # Check feedback quality (presence of reasoning)
        has_good_feedback = any(
            step.get("action", {}).get("reasoning") and 
            len(step.get("action", {}).get("reasoning", "")) > 20
            for step in trajectory
        )
        
        # Calculate score
        bug_score = (bugs_found / total_bugs) * 0.45 if total_bugs > 0 else 0.0
        feedback_score = 0.3 if has_good_feedback else 0.0
        decision_score = 0.25 if correct_decision else 0.0
        
        final_score = bug_score + feedback_score + decision_score
        return max(0.0, min(1.0, final_score))


class ContentModerationGrader(TaskGrader):
    """Grader for content moderation task."""
    
    def grade(self, trajectory: List[Dict[str, Any]], final_state: Dict[str, Any]) -> float:
        """
        Evaluate content moderation accuracy.
        
        Scoring:
        - Correct harmful content detection: +0.35
        - False positive prevention: +0.35 (penalize incorrectly flagging safe content)
        - Consistency: +0.3 if consistent policy application across similar items
        """
        correct_detections = 0
        false_positives = 0
        total_harmful = final_state.get("total_harmful_items", 5)
        total_safe = final_state.get("total_safe_items", 5)
        
        # Track detections and false positives
        for step in trajectory:
            action = step.get("action", {})
            if action.get("action_type") == "flag_content":
                target_id = action.get("target_id")
                item = final_state.get("items", {}).get(target_id, {})
                
                if item.get("is_harmful"):
                    correct_detections += 1
                else:
                    false_positives += 1
        
        # Calculate precision and recall
        detection_rate = (correct_detections / total_harmful) if total_harmful > 0 else 0.0
        false_positive_rate = (false_positives / total_safe) if total_safe > 0 else 0.0
        
        # Score components
        detection_score = detection_rate * 0.35
        fp_score = max(0.0, (1.0 - false_positive_rate) * 0.35)
        consistency_score = 0.3  # Assume consistency if grading logic is followed
        
        final_score = detection_score + fp_score + consistency_score
        return max(0.0, min(1.0, final_score))
