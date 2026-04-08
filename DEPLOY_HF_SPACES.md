# Deploy to Hugging Face Spaces

## Quick Setup

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces):
   - Click "Create new Space"
   - Choose **Docker** as the runtime
   - Fill in name and visibility settings

2. Clone the Space repository:
```bash
git clone https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME
cd YOUR-SPACE-NAME
```

3. Copy all files from this repository:
```bash
cp /path/to/meta/Dockerfile .
cp /path/to/meta/requirements.txt .
cp /path/to/meta/app.py .
cp /path/to/meta/inference.py .
cp -r /path/to/meta/openenv .
```

4. Commit and push:
```bash
git add .
git commit -m "Deploy MetaOpenEnv"
git push
```

5. Monitor the build on HF Spaces dashboard
   - Build will start automatically
   - Check logs for any errors
   - Once complete, your Space will be live!

## Testing Your Deployment

Once deployed, access the API:

```bash
# View API docs (interactive)
https://your-username-your-space.hf.space/docs

# Test /reset endpoint
curl -X POST https://your-username-your-space.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_type":"email_triage"}'

# Test /step endpoint
curl -X POST https://your-username-your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"classify","target_id":"email_0","classification":"work"}'
```

## Environment Variables

Set these in your HF Space settings:
- `HF_TOKEN`: Your Hugging Face token (optional)
- `PORT`: Automatically set to 7860

## Available Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state` - Get current state
- `GET /config` - Get configuration
- `POST /evaluate` - Evaluate episode
- `GET /docs` - Interactive API documentation (Swagger UI)

## Files to Push

Make sure these files are in your HF Space repo:
- ✅ `Dockerfile` - Container configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `app.py` - FastAPI application
- ✅ `inference.py` - Baseline inference
- ✅ `openenv/` - Core environment module
- ✅ `README.md` - Documentation

That's it! Your environment is now deployed on HF Spaces and will run automatically.
