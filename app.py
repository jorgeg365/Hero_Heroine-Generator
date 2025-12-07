import torch
import gradio as gr
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageEnhance, ImageFilter
import traceback

# 1) Load the pipeline (SD 1.5 img2img)
model_id = "runwayml/stable-diffusion-v1-5"

print("Loading Stable Diffusion model... This may take a few minutes on first run.")
try:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # If you have a CUDA GPU, send pipeline to GPU
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA GPU not found. This will be VERY slow on CPU.")
        pipe = pipe.to("cpu")
    
    # Use DPMSolverMultistepScheduler for better quality with optimized settings
    scheduler_config = pipe.scheduler.config
    scheduler_config["use_karras_sigmas"] = True  # Better quality sampling
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    
    # Disable attention/VAE slicing for maximum GPU speed (RTX 4070 Ti has enough VRAM)
    # These are only needed if you run out of memory
    # if torch.cuda.is_available():
    #     pipe.enable_attention_slicing()
    #     pipe.enable_vae_slicing()
    
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    pipe = None


# 2) Define base prompt parts - Enhanced for fully rendered quality
BASE_PROMPT = (
    "masterpiece, best quality, ultra high quality, extremely detailed, "
    "fully rendered, complete rendering, professional photography, "
    "8k uhd, dslr, soft lighting, high quality, film grain, "
    "cinematic portrait, comic-book style, sharp focus, "
    "vibrant colors, superhero costume, dramatic lighting, ultra realistic, "
    "photorealistic, detailed skin texture, detailed eyes, detailed hair, "
    "perfect face, symmetrical face, beautiful face, detailed face, "
    "high resolution, 4k, sharp details, professional lighting, "
    "fully detailed costume, complete outfit, detailed background, "
    "polished, refined, finished artwork, no artifacts, clean rendering, "
    "detailed textures, depth of field, volumetric lighting, "
    "cinematic composition, professional grade, studio quality"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, lowres, bad anatomy, bad hands, text, error, "
    "missing fingers, extra digit, fewer digits, cropped, worst quality, "
    "low quality, normal quality, jpeg artifacts, signature, watermark, "
    "username, artist name, distorted face, extra limbs, deformed, "
    "ugly, duplicate, morbid, mutilated, out of frame, mutation, "
    "bad proportions, gross proportions, malformed limbs, mutated hands, "
    "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, "
    "bad art, bad anatomy, bad proportions, cloned face, disfigured, "
    "oversaturated, undersaturated, grainy, unfinished, incomplete, "
    "unrendered, low detail, missing details, artifacts, noise, "
    "compression artifacts, pixelated, low resolution, sketchy, "
    "rough draft, work in progress, unfinished rendering"
)

HERO_STYLE_MAP = {
    "Tech Hero": (
        "high-tech cyber armor, fully detailed armor pieces, glowing circuits, "
        "LED lights, holographic displays, futuristic city skyline, neon signs, "
        "cyberpunk aesthetic, metallic textures, reflective surfaces, "
        "complete tech suit, detailed mechanical parts, advanced technology"
    ),
    "Mystic Hero": (
        "enchanted robes, fully detailed magical garments, glowing runes, "
        "magical aura, energy effects, ancient temple background, mystical atmosphere, "
        "magical particles, spell effects, detailed mystical symbols, "
        "complete magical outfit, ethereal lighting, fantasy elements"
    ),
    "Cosmic Hero": (
        "cosmic armor, fully detailed space suit, nebula background, "
        "stars and galaxies, floating debris, space environment, "
        "cosmic energy, stellar effects, detailed space textures, "
        "complete cosmic outfit, celestial lighting, astronomical details"
    ),
    "Street Vigilante": (
        "urban street clothes with fully detailed armor pieces, tactical gear, "
        "city rooftops at night, neon lights, urban environment, "
        "detailed street wear, complete vigilante outfit, tactical equipment, "
        "night city atmosphere, detailed urban background, street aesthetic"
    )
}


# 3) Generation function
def make_hero(input_image: Image.Image, hero_style: str, strength: float, guidance: float):
    try:
        if pipe is None:
            return None, "Error: Model not loaded. Please check the terminal for errors."
        
        if input_image is None:
            return None, "Please upload an image first."

        # Build prompt with explicit ethnicity-friendly instruction and face preservation
        style_text = HERO_STYLE_MAP.get(hero_style, "")
        prompt = (
            f"{BASE_PROMPT}, {style_text}, "
            "keep original person's facial features, exact face, same person, "
            "same skin tone, same ethnicity as the input image, preserve identity, "
            "maintain facial structure, keep original eyes, keep original nose, "
            "keep original mouth, same hair color, same hair style, "
            "fully rendered character, complete character design, "
            "detailed full body, complete outfit rendering, professional character art"
        )

        # Convert and resize - using 768x768 for better quality (if GPU) or 512x512 for CPU
        input_image = input_image.convert("RGB")
        
        # Use higher resolution for better quality if GPU available
        if torch.cuda.is_available():
            # RTX 4070 Ti - using 768x768 for good balance of quality and speed
            target_size = (768, 768)
        else:
            # CPU: stick with 512x512 for speed
            target_size = (512, 512)
        
        # Maintain aspect ratio while resizing for better quality
        input_image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create a new image with the target size and paste the resized image
        new_image = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_size[0] - input_image.width) // 2
        paste_y = (target_size[1] - input_image.height) // 2
        new_image.paste(input_image, (paste_x, paste_y))
        input_image = new_image

        print(f"Generating hero image with style: {hero_style}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Balance quality vs speed
        # More steps = better quality but slower
        if torch.cuda.is_available():
            num_steps = 30  # GPU: fast and high quality (optimized for speed)
        else:
            num_steps = 25  # CPU: reasonable quality
        
        print(f"Running {num_steps} inference steps...")
        
        # Disable autocast on CPU (it can cause issues)
        if torch.cuda.is_available():
            with torch.autocast("cuda"):
                result = pipe(
                    prompt=prompt,
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=num_steps,
                )
        else:
            # CPU mode - no autocast
            result = pipe(
                prompt=prompt,
                image=input_image,
                strength=strength,
                guidance_scale=guidance,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=num_steps,
            )

        print("Generation complete!")
        
        # Post-process for better quality and fully rendered appearance
        output_image = result.images[0]
        
        # Enhanced post-processing for fully rendered quality
        # Enhance sharpness for crisp details
        enhancer = ImageEnhance.Sharpness(output_image)
        output_image = enhancer.enhance(1.2)  # Increased for fully rendered look
        
        # Enhance contrast for depth
        enhancer = ImageEnhance.Contrast(output_image)
        output_image = enhancer.enhance(1.08)
        
        # Enhance color saturation slightly for vibrant rendering
        enhancer = ImageEnhance.Color(output_image)
        output_image = enhancer.enhance(1.05)
        
        # Apply unsharp mask for better detail rendering
        output_image = output_image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=3))
        
        # Additional sharpening pass for fully rendered appearance
        output_image = output_image.filter(ImageFilter.SHARPEN)
        
        return output_image, None
        
    except torch.cuda.OutOfMemoryError:
        error_msg = "Out of GPU memory! Try reducing image size or closing other applications."
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return None, error_msg


# 4) Build Gradio UI
def build_demo():
    with gr.Blocks(title="Superhero/ Heroine Generator") as demo:
        gr.Markdown(
            """
            # 🦸‍♂️ Superhero / Heroine Generator

            Upload a clear photo of a person, and choose a hero style.
            The model will try to keep the face and skin tone similar,
            while changing outfit and overall vibe.

            > Tip: Use a front-facing, well-lit photo for best results.
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload your photo",
                    type="pil"
                )
                hero_style = gr.Radio(
                    choices=list(HERO_STYLE_MAP.keys()),
                    value="Tech Hero",
                    label="Hero Style"
                )
                strength = gr.Slider(
                    minimum=0.3,
                    maximum=0.9,
                    value=0.55,
                    step=0.05,
                    label="Transformation strength (lower = more like original, better face preservation)"
                )
                guidance = gr.Slider(
                    minimum=4.0,
                    maximum=15.0,
                    value=9.0,
                    step=0.5,
                    label="Prompt guidance scale (higher = follows prompt more closely)"
                )
                generate_btn = gr.Button("Generate Hero", variant="primary")

            with gr.Column():
                output_image = gr.Image(
                    label="Hero version",
                    type="pil"
                )
                error_output = gr.Textbox(
                    label="Status",
                    visible=True,
                    interactive=False
                )

        def generate_with_error_handling(img, style, str_val, guid_val):
            device_info = "GPU" if torch.cuda.is_available() else "CPU"
            if torch.cuda.is_available():
                status_msg = "⏳ Generating high-quality image on GPU (30 steps, 768x768)... This may take 15-30 seconds..."
            else:
                status_msg = "⏳ Generating high-quality image on CPU... This may take 5-8 minutes. Please be patient..."
            
            try:
                result, error = make_hero(img, style, str_val, guid_val)
                if error:
                    return result, f"❌ {error}"
                return result, "✅ Generation complete! Download your hero image above."
            except Exception as e:
                return None, f"❌ Unexpected error: {str(e)}"

        generate_btn.click(
            fn=generate_with_error_handling,
            inputs=[input_image, hero_style, strength, guidance],
            outputs=[output_image, error_output]
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    # For Hugging Face Spaces, use share=False and let Spaces handle the server
    # For local use, specify server settings
    import os
    if os.getenv("SPACE_ID"):  # Running on Hugging Face Spaces
        demo.launch(share=False)
    else:  # Running locally
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            inbrowser=True
        )
