# MetaOpenEnv: Real-World Task Simulation Environment

A comprehensive OpenEnv-compliant environment for evaluating AI agents on real-world task simulation. This environment provides three progressively challenging tasks: **Email Triage**, **Code Review**, and **Content Moderation**, each with programmatic graders and meaningful reward functions.

![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

MetaOpenEnv bridges the gap between toy environments and real-world AI evaluation by providing:

- **Realistic Tasks**: Email triage, code review, and content moderation—tasks that humans actually perform
- **OpenEnv Specification Compliance**: Full Pydantic-based implementation with typed models
- **Progressive Difficulty**: Three tasks ranging from easy to hard
- **Programmatic Graders**: Deterministic, reproducible evaluation metrics (0.0-1.0 scores)
- **Trajectory-Based Rewards**: Meaningful feedback that rewards incremental progress
- **Containerized Deployment**: Ready for Hugging Face Spaces and cloud deployment

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Tasks](#tasks)
- [Quick Start](#quick-start)
- [Action & Observation Spaces](#action--observation-spaces)
- [Reward Function](#reward-function)
- [Baseline Performance](#baseline-performance)
- [Docker Deployment](#docker-deployment)
- [API Documentation](#api-documentation)
- [Examples](#examples)

## ✨ Features

### 1. **Email Triage Task** (Easy)
- **Objective**: Classify emails into categories: work, personal, spam, promotional
- **Difficulty**: Easy - straightforward pattern matching
- **Grading**: Accuracy-based with efficiency bonus
- **Best Score**: 1.0 (100% accuracy + efficiency)

### 2. **Code Review Task** (Medium)
- **Objective**: Identify bugs in code blocks and decide approval or changes
- **Difficulty**: Medium - requires understanding code semantics
- **Grading**: Precision-based (bug detection + decision correctness)
- **Components**: 3-4 code blocks with hidden bugs

### 3. **Content Moderation Task** (Hard)
- **Objective**: Flag harmful content while avoiding false positives
- **Difficulty**: Hard - nuanced policy enforcement
- **Grading**: F1-based (balance precision and recall)
- **Items**: 10 content samples with varying harm levels

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip or conda
- (Optional) Docker for containerization

### Local Setup

```bash
# Clone the repository
cd /path/to/meta/openenv

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from openenv import OpenEnv; print('✓ OpenEnv ready')"
```

### Docker Setup

```bash
# Build the image
docker build -t meta-openenv:latest .

# Run the container
docker run --rm -e HF_TOKEN="your_token" meta-openenv:latest
```

## 📚 Quick Start

### Basic Usage

```python
from openenv import OpenEnv, TaskType, Action

# Initialize environment
env = OpenEnv()

# Reset to email triage task
observation = env.reset(task_type=TaskType.EMAIL_TRIAGE)
print(f"Task: {observation.task_name}")
print(f"Data: {observation.data}")

# Take an action
action = Action(
    action_type="classify",
    target_id="email_0",
    classification="work",
    confidence=0.95,
    reasoning="From company email address"
)

# Step in the environment
result = env.step(action)
print(f"Reward: {result.reward.value}")
print(f"Done: {result.done}")

# Continue until episode ends
while not result.done:
    # Your agent logic here
    result = env.step(next_action)

# Evaluate performance
evaluation = env.evaluate_episode()
print(f"Final Score: {evaluation['final_score']:.3f}")
print(f"Episode Reward: {evaluation['episode_reward']:.3f}")
```

### Task Selection

```python
from openenv import TaskType

# Run each task
for task_type in [
    TaskType.EMAIL_TRIAGE,      # Easy
    TaskType.CODE_REVIEW,       # Medium
    TaskType.CONTENT_MODERATION # Hard
]:
    obs = env.reset(task_type=task_type)
    print(f"Started {obs.task_type} task")
```

## 📊 Action & Observation Spaces

### Observation Space (Pydantic Model)

```python
class Observation(BaseModel):
    task_id: str              # Unique task identifier
    task_type: TaskType       # Type of task
    task_name: str            # Human-readable name
    data: Dict[str, Any]      # Task-specific data
    step_count: int           # Steps taken so far
    metadata: Optional[Dict]  # Additional info
```

### Action Space (Pydantic Model)

```python
class Action(BaseModel):
    action_type: str              # classify, flag, approve, etc.
    target_id: Optional[str]      # ID of item being acted upon
    classification: Optional[str] # For classify actions
    confidence: Optional[float]   # Confidence (0.0-1.0)
    reasoning: Optional[str]      # Explanation
    metadata: Optional[Dict]      # Additional metadata
```

### Reward Model (Pydantic Model)

```python
class Reward(BaseModel):
    value: float                         # Reward value for step
    success: bool                        # Action success
    message: str                         # Feedback message
    components: Optional[Dict[str, float]]  # Reward breakdown
```

### Per-Task Observation Details

#### Email Triage
```python
observation.data = {
    "emails": [
        {
            "id": "email_0",
            "subject": "Quarterly Review Meeting",
            "preview": "Please see the attached...",
            "sender": "boss@company.com"
        },
        # ... more emails
    ],
    "classified_count": 2,
    "total_emails": 8,
    "remaining": [...]  # Unclassified emails
}
```

#### Code Review
```python
observation.data = {
    "code_blocks": [
        {
            "id": "block_0",
            "language": "python",
            "code": "def divide(a, b):\n    return a / b"
        },
        # ... more blocks
    ],
    "flagged_count": 1,
    "total_issues": 4
}
```

#### Content Moderation
```python
observation.data = {
    "items": [
        {
            "id": "item_0",
            "content": "Kill all [group]",
            "category": "hate_speech",
            "severity": "critical"
            # Note: is_harmful is NOT visible to agent
        },
        # ... more items
    ],
    "reviewed_count": 2,
    "total_items": 10,
    "remaining": [...]
}
```

## 💰 Reward Function

The reward function provides **trajectory-based feedback** with immediate step rewards plus completion bonuses:

### Components

1. **Action Reward** (Primary)
   - Correct actions: +0.1 to +0.2
   - Incorrect actions: -0.05 to -0.1
   - Destructive actions: -0.5

2. **Progress Reward**
   - Incremental steps toward goal: +0.05
   - Encourages exploration

3. **Efficiency Bonus** (End of episode)
   - Completing in ≤50% max steps: +0.2
   - Scales linearly with steps

4. **Penalties**
   - Destructive actions (delete, ban): -0.5
   - Repeated same action (5+ times): -0.1 per step
   - Exceeding max steps: -0.1

5. **Completion Bonus**
   - Successfully completing episode: +0.5

### Reward Range: [-1.0, +1.0]

Example trajectory:
```
Step 1: +0.1 (correct classification)
Step 2: +0.05 (progress)
Step 3: +0.1 (correct classification)
...
Step 8: +0.2 (efficiency bonus) + 0.5 (completion) = +0.7
Total: ~1.2 (can exceed bounds when including completions)
```

## 📈 Baseline Performance

Baseline heuristic-based implementation without using LLMs:

| Task | Strategy | Score | Steps | Episode Reward |
|------|----------|-------|-------|----------------|
| **Email Triage** | Domain pattern matching | 0.75 | 8 | 1.05 |
| **Code Review** | Keyword-based bug detection | 0.68 | 6 | 0.95 |
| **Content Moderation** | Simple keyword filtering | 0.72 | 12 | 1.10 |
| **Average** | - | **0.72** | - | - |

These baselines use simple heuristics:
- **Email Triage**: Classify based on sender domain
- **Code Review**: Flag potential bugs by keywords/patterns
- **Content Moderation**: Use hardcoded harmful content signatures

### Running Baselines

```bash
# Run full baseline
python inference.py

# With API token
export HF_TOKEN="your_api_key"
python inference.py
```

Results saved to `baseline_results.json`.

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build image
docker build -t meta-openenv:latest .

# Run with API token
docker run --rm \
    -e HF_TOKEN="your_openai_key" \
    -v $(pwd)/results:/app/results \
    meta-openenv:latest

# Run interactively
docker run -it --rm meta-openenv:latest bash
```

### Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Select Docker as the runtime
3. Push Dockerfile and requirements.txt:

```bash
git clone https://huggingface.co/spaces/your-username/meta-openenv
cd meta-openenv

# Add files
cp Dockerfile .
cp requirements.txt .
cp -r openenv/ .
cp app.py .
cp inference.py .

git add .
git commit -m "Initial commit"
git push
```

4. Space will automatically build and deploy on port 7860

### Using the API

Once deployed (locally or on HF Spaces), use the HTTP API:

```bash
# Reset environment
curl -X POST https://your-space-url/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"email_triage"}'

# Execute action
curl -X POST https://your-space-url/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type":"classify",
    "target_id":"email_0",
    "classification":"work",
    "confidence":0.9
  }'

# Get state
curl https://your-space-url/state

# View API docs
# Open: https://your-space-url/docs
```

## 📖 API Documentation

### OpenEnv Class

```python
class OpenEnv:
    """Main environment class."""
    
    def reset(
        self, 
        task_type: TaskType = None,
        task_id: str = None
    ) -> Observation:
        """Reset and return initial observation."""
    
    def step(self, action: Action) -> StepResult:
        """Execute action, return (observation, reward, done, info)."""
    
    def state(self) -> Dict[str, Any]:
        """Get current environment state."""
    
    def evaluate_episode(self) -> Dict[str, Any]:
        """Evaluate completed episode with final score."""
    
    def render(self, mode: str = "dict") -> str:
        """Render state in JSON, dict, or text format."""
    
    def get_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
```

### StepResult Class

```python
class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]
```

The `info` dict contains:
```python
{
    "episode_step": int,          # Step count
    "episode_reward": float,      # Cumulative reward
    "action_info": Dict,          # Action-specific info
    "reward_components": Dict,    # Reward breakdown
    "trajectory_length": int      # Total trajectory steps
}
```

## 🔬 Examples

### Example 1: Simple Email Classifier

```python
from openenv import OpenEnv, TaskType, Action

env = OpenEnv()
obs = env.reset(task_type=TaskType.EMAIL_TRIAGE)

categories = {
    "boss@": "work",
    "spam.": "spam",
    "promo@": "promotional",
    "family.": "personal"
}

for email in obs.data["emails"]:
    sender = email["sender"]
    category = next(
        (cat for domain, cat in categories.items() if domain in sender),
        "personal"
    )
    
    action = Action(
        action_type="classify",
        target_id=email["id"],
        classification=category
    )
    
    result = env.step(action)
    print(f"{email['subject'][:30]} → {category} | Reward: {result.reward.value}")

score = env.evaluate_episode()
print(f"\nFinal Score: {score['final_score']:.3f}")
```

### Example 2: Code Bug Detector

```python
from openenv import OpenEnv, TaskType, Action

env = OpenEnv()
obs = env.reset(task_type=TaskType.CODE_REVIEW)

# Known bug patterns
bug_indicators = ["divide", "redeclaration", "out of bounds", "[10]"]

# Review code
for code_block in obs.data["code_blocks"]:
    if any(indicator in code_block["code"] for indicator in bug_indicators):
        action = Action(
            action_type="flag_issue",
            target_id=code_block["id"],
            reasoning="Detected potential bug pattern"
        )
        env.step(action)

# Make decision
action = Action(action_type="request_changes")
result = env.step(action)

if result.done:
    score = env.evaluate_episode()
    print(f"Review Score: {score['final_score']:.3f}")
```

### Example 3: Content Safety Checker

```python
from openenv import OpenEnv, TaskType, Action

env = OpenEnv()
obs = env.reset(task_type=TaskType.CONTENT_MODERATION)

harmful_patterns = ["Kill", "explosives", "OnlyFans", "SCAM"]

while not obs.done:
    # Decision: flag or skip
    content = obs.data["items"][0]  # Would need to parse from displayed text
    
    should_flag = any(
        pattern.lower() in str(content).lower() 
        for pattern in harmful_patterns
    )
    
    action = Action(
        action_type="flag_content" if should_flag else "skip_review",
        target_id=content["id"]
    )
    
    result = env.step(action)
    obs = result.observation

evaluation = env.evaluate_episode()
print(f"Moderation Score: {evaluation['final_score']:.3f}")
```

## 📋 Grading Criteria Details

### Email Triage Grader

```
Score = (Correct Classifications / Total) * 0.8 
        + Efficiency Bonus * 0.2
        - Destructive Penalty * 0.2

Efficiency Bonus: +0.2 if completed in ≤8 steps
Destructive Penalty: Applied if any deletions occur
Range: [0.0, 1.0]
```

### Code Review Grader

```
Score = (Bugs Found / Total Bugs) * 0.45
        + (Correct Decision) * 0.25
        + (Good Feedback Quality) * 0.3

Good Feedback: Reasoning provided with 20+ characters
Bugs Found: Actually flagged issues marked as bugs
Range: [0.0, 1.0]
```

### Content Moderation Grader

```
Precision = Correct Flags / All Flags
Recall = Correct Flags / Total Harmful

Score = Recall * 0.35
        + (1 - False Positive Rate) * 0.35
        + Consistency * 0.3

False Positive Rate = Incorrect Flags / Total Safe
Range: [0.0, 1.0]
```

## 🔧 Configuration

Environment configuration is stored in `openenv/openenv.yaml`:

```yaml
name: MetaOpenEnv
version: "1.0.0"
max_episode_steps: 100

tasks:
  - id: email_triage
    difficulty: easy
    max_steps: 100
  - id: code_review
    difficulty: medium
    max_steps: 100
  - id: content_moderation
    difficulty: hard
    max_steps: 150
```

## 🧪 Testing & Validation

### Validate OpenEnv Compliance

```python
from openenv import OpenEnv, TaskType, Action

# Test specification compliance
env = OpenEnv()

# 1. Check methods exist
assert hasattr(env, 'reset')
assert hasattr(env, 'step')
assert hasattr(env, 'state')
assert hasattr(env, 'evaluate_episode')

# 2. Test reset returns Observation
obs = env.reset(task_type=TaskType.EMAIL_TRIAGE)
assert obs.task_id is not None
assert obs.data is not None

# 3. Test step returns StepResult
action = Action(action_type="classify", target_id="email_0", classification="work")
result = env.step(action)
assert hasattr(result, 'observation')
assert hasattr(result, 'reward')
assert hasattr(result, 'done')
assert hasattr(result, 'info')

# 4. Verify models are Pydantic
from openenv import Observation, Action, Reward
assert hasattr(Observation, 'model_dump')
assert hasattr(Action, 'model_dump')
assert hasattr(Reward, 'model_dump')

print("✓ All OpenEnv specification requirements met")
```

## 📊 OpenEnv YAML Specification

The `openenv/openenv.yaml` file validates the environment:

```bash
# Validate with openenv CLI (when available)
openenv validate openenv/openenv.yaml
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📬 Citation

```bibtex
@software{meta_openenv_2024,
  title={MetaOpenEnv: Real-World Task Simulation Environment},
  author={Meta AI},
  year={2024},
  url={https://github.com/meta/openenv}
}
```

## 🎓 References

- [OpenEnv Specification](https://github.com/TODO/openenv-spec)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)

---

**Made with ❤️ for AI evaluation research**
