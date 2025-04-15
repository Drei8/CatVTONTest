import os
import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch

from pyngrok import ngrok
from rembg import remove
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/tryon": {"origins": "*"}})  # Use specific origin in prod


# Configuration
BASE_MODEL_PATH = "booksforcharlie/stable-diffusion-inpainting"
RESUME_PATH = "zhengchong/CatVTON"
OUTPUT_DIR = "resource/demo/output"
WIDTH, HEIGHT = 768, 1024

# Load model components
repo_path = snapshot_download(repo_id=RESUME_PATH)

pipeline = CatVTONPipeline(
    base_ckpt=BASE_MODEL_PATH,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype("bf16"),
    use_tf32=True,
    device='cuda'
)

mask_processor = VaeImageProcessor(
    vae_scale_factor=8,
    do_normalize=False,
    do_binarize=True,
    do_convert_grayscale=True
)

automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda'
)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def decode_base64_to_image(b64_string):
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def remove_background(image):
    image_rgba = image.convert("RGBA")
    removed = remove(image_rgba)
    return removed.convert("RGB")

@app.route("/", methods=["GET"])
def home():
    return "CatVTON Flask API is running with base64 and cloth background removal!"

@app.route("/tryon", methods=["POST"])
def tryon():
    try:
        data = request.get_json()

        # Required base64 inputs
        person_b64 = data.get("person_image")
        cloth_b64 = data.get("cloth_image")
        cloth_type = data.get("cloth_type", "upper")

        if not person_b64 or not cloth_b64:
            return jsonify({"status": "error", "message": "Missing image data"}), 400

        # Optional params
        num_inference_steps = int(data.get("num_inference_steps", 100))
        guidance_scale = float(data.get("guidance_scale", 2.5))
        seed = int(data.get("seed", 42))

        # Decode base64
        person_img = decode_base64_to_image(person_b64)
        cloth_img = decode_base64_to_image(cloth_b64)

        person_img = person_img.rotate(-90, expand=True)

        # Resize person image
        person_img = resize_and_crop(person_img, (WIDTH, HEIGHT))

        # 🔥 Remove background from cloth image before resize
        cloth_img = remove_background(cloth_img)
        cloth_img = resize_and_padding(cloth_img, (WIDTH, HEIGHT))

        # Generate mask
        mask = automasker(person_img, cloth_type)['mask']
        mask = mask_processor.blur(mask, blur_factor=9)

        # Setup generator
        generator = torch.Generator(device='cuda').manual_seed(seed) if seed != -1 else None

        # Run try-on
        result_img = pipeline(
            image=person_img,
            condition_image=cloth_img,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )[0]

        # Create visual grid
        masked_person = vis_mask(person_img, mask)
        final_image = image_grid([person_img, masked_person, cloth_img, result_img], 1, 4)

        # Save result
        date_str = datetime.now().strftime("%Y%m%d%H%M%S")
        folder_path = os.path.join(OUTPUT_DIR, date_str[:8])
        os.makedirs(folder_path, exist_ok=True)
        save_path = os.path.join(folder_path, f"{date_str[8:]}.png")
        final_image.save(save_path)

        # Encode to base64
        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        result_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "status": "success",
            "result_image": result_b64,
            "saved_to": save_path
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=8888)


