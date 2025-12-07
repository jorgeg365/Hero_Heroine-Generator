# Quick GitHub Setup Guide

## Step 1: Initialize Git Repository

Open PowerShell in your project folder (`F:\Face Swap`) and run:

```powershell
git init
git add .
git commit -m "Initial commit: Superhero Generator App"
```

## Step 2: Create GitHub Repository

1. Go to: https://github.com/new
2. **Repository name:** `superhero-generator` (or your choice)
3. **Description:** "Transform photos into superhero portraits using Stable Diffusion"
4. **Visibility:** Public (required for free Hugging Face GPU)
5. **DO NOT** check "Initialize with README" (you already have files)
6. Click **"Create repository"**

## Step 3: Connect and Push

GitHub will show you commands. Use these (replace YOUR_USERNAME):

```powershell
git remote add origin https://github.com/YOUR_USERNAME/superhero-generator.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy to Hugging Face Spaces

1. Go to: https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Space name:** `superhero-generator`
   - **SDK:** `Gradio`
   - **Hardware:** `GPU (T4 small)` - FREE!
   - **Visibility:** `Public`
4. Click **"Create Space"**
5. In Space settings → **Repository** → **Add repository**
6. Select your GitHub repo
7. Wait 5-10 minutes for deployment

## Your App Will Be Live At:

`https://huggingface.co/spaces/YOUR_USERNAME/superhero-generator`

## Notes

- ✅ **Free GPU** on Hugging Face Spaces (T4)
- ✅ **Automatic updates** when you push to GitHub
- ✅ **Public URL** to share with others
- ❌ **Vercel won't work** - no GPU support

