 HEAD
# Hero_Heroine-Generator

# 🦸‍♂️ Superhero / Heroine Generator

Transform your photos into superhero/heroine portraits using Stable Diffusion while preserving your ethnicity and facial features.

## Prerequisites

Before starting, make sure you have:

- **Python 3.10 or 3.11** installed
- **Git** installed
- **Latest NVIDIA drivers** (if using GPU)
- **At least 8-12 GB VRAM** recommended for smooth operation (SD 1.5 is used, which is lighter than SDXL)

## Setup Instructions

### Step 1: Create Virtual Environment

Open a terminal in Cursor IDE (Ctrl+`) and run:

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

Your prompt should now show `(venv)` at the beginning.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will download PyTorch, diffusers, Gradio, and other required packages. The first install may take several minutes.

### Step 4: Run the Application

```bash
python app.py
```

The first run will download the `runwayml/stable-diffusion-v1-5` model (approximately 4GB). After loading, Gradio will print a URL like:

```
Running on local URL:  http://127.0.0.1:7860
```

Open that URL in your browser to use the app.

## How to Use

1. **Upload a photo**: Use a clear, front-facing photo with good lighting for best results
2. **Choose a hero style**: 
   - Tech Hero
   - Mystic Hero
   - Cosmic Hero
   - Street Vigilante
3. **Adjust settings**:
   - **Strength**: Lower values (0.3-0.5) keep more of the original, higher values (0.7-0.9) transform more
   - **Guidance**: Controls how strongly the model follows the text prompt (7.5 is a good default)
4. **Click "Generate Hero"** and wait for the result

## Tips for Better Results

- **If your face changes too much**: Lower the strength slider (try 0.4-0.5)
- **For better identity preservation**: The prompt already includes instructions to keep facial features and skin tone
- **Best input photos**: Front-facing, well-lit, clear face visible

## Troubleshooting

### CUDA/GPU Issues
If you see CUDA errors:
- Make sure your NVIDIA drivers are up to date
- Verify CUDA is installed: `nvidia-smi` in terminal
- The app will fall back to CPU if GPU is not available (but will be very slow)

### Out of Memory Errors
If you run out of VRAM:
- The code already resizes images to 512x512 to reduce memory usage
- You can reduce `num_inference_steps` in `app.py` (currently 30, try 20)
- Consider using a smaller model or running on CPU

### Model Download Issues
If the model download fails:
- Check your internet connection
- The model downloads to `~/.cache/huggingface/` by default
- You may need to manually download and place it in the cache folder

## Next Steps (Optional Enhancements)

For even stronger identity preservation, you can later add:
- **IP-Adapter** or **InstantID** integration
- These models lock in facial features much more tightly while still allowing style changes

## Project Structure

```
hero-generator/
  app.py              # Main application with Gradio UI
  requirements.txt    # Python dependencies
  README.md          # This file
  venv/              # Virtual environment (created after setup)
```

 3814266 (Initial commit: Hero Heroine Generator)
