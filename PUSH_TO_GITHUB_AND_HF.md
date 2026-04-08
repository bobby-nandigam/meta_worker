# GitHub → HF Spaces Push Guide

## Step 1: Create GitHub Repository

```bash
# Go to https://github.com/new and create a repo called "meta_worker"
# Then locally:

cd /Users/bobbynandigam/Desktop/project/meta

# Initialize git if not already
git init
git add .
git commit -m "Initial OpenEnv environment commit"

# Add GitHub remote
git remote add github https://github.com/YOUR_USERNAME/meta_worker.git
git branch -M main
git push -u github main
```

## Step 2: Push to HF Spaces

```bash
# Clone HF Space repo (use your HF token as password when prompted)
git clone https://huggingface.co/spaces/bobbynandigam/meta_worker
cd meta_worker

# Copy files from local project
cp -r /Users/bobbynandigam/Desktop/project/meta/* .
cp -r /Users/bobbynandigam/Desktop/project/meta/.gitignore .

# Add HF remote
git remote add huggingface https://huggingface.co/spaces/bobbynandigam/meta_worker

# Commit and push
git add .
git commit -m "Deploy MetaOpenEnv to HF Spaces"
git push huggingface main
```

## Quick Commands (Copy-Paste Ready)

### Push to GitHub:
```bash
cd /Users/bobbynandigam/Desktop/project/meta
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/meta_worker.git
git branch -M main
git push -u origin main
```

### Push to HF Spaces:
```bash
# Create temp directory
mkdir -p /tmp/hf_space
cd /tmp/hf_space

# Clone HF Space (replace YOUR_TOKEN with actual HF token)
git clone https://huggingface.co/spaces/bobbynandigam/meta_worker

# Copy files
cp -r /Users/bobbynandigam/Desktop/project/meta/* meta_worker/
cd meta_worker

# Push to HF
git add .
git commit -m "Deploy MetaOpenEnv"
git push
```

## Generate HF Access Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `hf_spaces_deployment`
4. Role: `write`
5. Copy token
6. When git prompts for password: paste the token

## Check Deployment Status

- **GitHub**: https://github.com/YOUR_USERNAME/meta_worker
- **HF Spaces**: https://huggingface.co/spaces/bobbynandigam/meta_worker
  - Click "Logs" tab to see build status
  - Wait for "✓ Build successful"

Done! Both repos will be synced with your code.
