# How to Start the App

## Quick Start (Easiest Method)

**Double-click:** `start_app.bat`

This will automatically start the app for you.

---

## Manual Start Method

1. **Open PowerShell or Command Prompt**

2. **Navigate to the project folder:**
   ```powershell
   cd "F:\Face Swap"
   ```

3. **Start the app:**
   ```powershell
   python app.py
   ```

4. **Wait for the app to load:**
   - You'll see "Loading Stable Diffusion model..."
   - Then "Model loaded on GPU: NVIDIA GeForce RTX 4070 Ti"
   - Finally "Running on local URL: http://127.0.0.1:7861"

5. **Open your browser:**
   - Go to: **http://127.0.0.1:7861**
   - Or it should open automatically

---

## Notes

- **First time:** Model download may take a few minutes (~4GB)
- **After first run:** Model is cached, so startup is much faster
- **To stop:** Press `Ctrl+C` in the terminal, or close the terminal window
- **Port:** The app runs on port 7861 (changed from 7860 to avoid conflicts)

---

## Troubleshooting

**If port 7861 is busy:**
- The app will try to find another available port
- Check the terminal for the actual URL

**If GPU not detected:**
- Make sure your NVIDIA drivers are up to date
- Run `nvidia-smi` to verify GPU is working

**If app won't start:**
- Make sure Python is installed and in your PATH
- Check that all dependencies are installed: `pip install -r requirements.txt`

