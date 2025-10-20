# Deploy to Hugging Face Spaces

Step-by-step guide to deploy this Streamlit app to Hugging Face Spaces.

## Quick Deploy (5 minutes)

### 1. Create a Hugging Face Account

Go to https://huggingface.co/join and create a free account.

### 2. Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **Name**: `hepatitis-c-predictor` (or your preferred name)
   - **License**: MIT
   - **SDK**: Streamlit
   - **Visibility**: Public (or Private)
4. Click "Create Space"

### 3. Prepare Your Space

**Important**: Hugging Face Spaces looks for a README.md in the root with YAML frontmatter. We've created `README_HF.md` for this purpose.

Before pushing, rename it:

```bash
# Backup your original README (optional)
cp README.md README_GITHUB.md

# Use the HF-specific README
cp README_HF.md README.md
```

Or manually copy the YAML frontmatter from `README_HF.md` to the top of your `README.md`:

```yaml
---
title: Hepatitis C Predictor
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.50.0"
app_file: app.py
pinned: false
license: mit
---
```

### 4. Push Your Code

#### Option A: Via Git (Recommended)

```bash
# Add Hugging Face as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/hepatitis-c-predictor

# Push your code
git push hf main
```

#### Option B: Via Web Interface

1. Go to your Space
2. Click "Files and versions"
3. Click "Add file" → "Upload files"
4. Upload all project files (app.py, requirements.txt, src/, etc.)
5. **Important**: Either use `README_HF.md` or add the YAML frontmatter to your README.md

### 5. Wait for Build

Hugging Face will automatically:
- Install dependencies from `requirements.txt`
- Download the dataset on first run
- Start your Streamlit app

Build typically takes 3-5 minutes.

### 6. Access Your App

Your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/hepatitis-c-predictor
```

## Configuration

The following files configure the Space:

### `.streamlit/config.toml`
Configures Streamlit settings (already created)

### `requirements.txt`
Lists all Python dependencies (already exists)

### `packages.txt` (Optional)
System packages if needed (already created, currently empty)

## Troubleshooting

### Build Fails

Check the "Logs" tab in your Space for errors. Common issues:

1. **Missing dependencies**: Add to `requirements.txt`
2. **Memory issues**: Reduce model size or use CPU-only torch
3. **Port conflicts**: HF Spaces uses port 7860 automatically

### Dataset Download Issues

If auto-download fails:
1. Manually upload `hepatitis_data.csv` to the Space
2. Place it in `data/raw/` directory

### App Doesn't Start

Ensure `app.py` is in the root directory and Streamlit is in requirements.txt.

## Advanced: Environment Variables

If you need API keys or secrets:

1. Go to Space Settings
2. Add Repository secrets
3. Access in code:
   ```python
   import os
   api_key = os.getenv("MY_API_KEY")
   ```

## Advanced: Custom Docker

If you need more control, you can use a custom Dockerfile:

1. Change SDK to "Docker"
2. Upload your `Dockerfile`
3. HF will build and run the container

## Updating Your Space

To update after deployment:

```bash
# Make changes locally
git add .
git commit -m "Update model/features"
git push hf main
```

The Space will automatically rebuild.

## Free Tier Limits

Hugging Face Spaces free tier includes:
- 16 GB RAM
- 2 CPU cores
- Persistent storage
- Automatic SSL
- Custom domains (for PRO)

For GPU access, upgrade to PRO (not needed for this project).

## Make it Public

To share your work:

1. Go to Space Settings
2. Set visibility to "Public"
3. Share the URL: `https://huggingface.co/spaces/YOUR_USERNAME/hepatitis-c-predictor`

## Add to README Badge

Add this badge to your main README.md:

```markdown
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/YOUR_USERNAME/hepatitis-c-predictor)
```

## Support

- HF Spaces Documentation: https://huggingface.co/docs/hub/spaces
- Community Forum: https://discuss.huggingface.co/
