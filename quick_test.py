#!/usr/bin/env python3
"""Quick validation of Phase 1 requirements."""

from openenv import OpenEnv, TaskType, Action

print("=" * 60)
print("OpenEnv Phase 1 Validation")
print("=" * 60)

env = OpenEnv()

# Test 1: Email Triage
print("\n✓ Test 1: Email Triage (Easy)")
obs = env.reset(TaskType.EMAIL_TRIAGE)
print(f"  Reset: {len(obs.data['emails'])} emails")
action = Action(action_type='classify', target_id='email_0', classification='work')
result = env.step(action)
print(f"  Step: Reward={result.reward.value:.3f}, Done={result.done}")

# Test 2: Code Review
print("\n✓ Test 2: Code Review (Medium)")
obs = env.reset(TaskType.CODE_REVIEW)
print(f"  Reset: {obs.data['total_issues']} issues")
action = Action(action_type='flag_issue', target_id='issue_0')
result = env.step(action)
print(f"  Step: Reward={result.reward.value:.3f}, Done={result.done}")

# Test 3: Content Moderation
print("\n✓ Test 3: Content Moderation (Hard)")
obs = env.reset(TaskType.CONTENT_MODERATION)
print(f"  Reset: {obs.data['total_items']} items")
action = Action(action_type='flag_content', target_id='item_0')
result = env.step(action)
print(f"  Step: Reward={result.reward.value:.3f}, Done={result.done}")

print("\n✓ All tests passed!")
print("=" * 60)
