import argparse
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def main():
    parser = argparse.ArgumentParser(description="Run CatVTON from CLI.")
    parser.add_argument("--person_image", required=True, type=str)
    parser.add_argument("--cloth_image", required=True, type=str)
    parser.add_argument("--cloth_type", required=True, choices=["upper", "lower", "overall"])
    parser.add_argument("--base_model_path", type=str, default="booksforcharlie/stable-diffusion-inpainting")
    parser.add_argument("--resume_path", type=str, default="zhengchong/CatVTON")
    parser.add_argument("--output_dir", type=str, default="resource/demo/output")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")

    args = parser.parse_args()

    repo_path = snapshot_download(repo_id=args.resume_path)
    pipeline = CatVTONPipeline(
        base_ckpt=args.base_model_path,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(args.mixed_precision),
        use_tf32=args.allow_tf32,
        device="cuda"
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
        device="cuda",
    )

    person_image = Image.open(args.person_image).convert("RGB")
    cloth_image = Image.open(args.cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    mask = automasker(person_image, args.cloth_type)["mask"]
    mask = mask_processor.blur(mask, blur_factor=9)

    generator = None
    if args.seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    )[0]

    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_path = os.path.join(args.output_dir, f"tryon_{date_str}.png")
    os.makedirs(args.output_dir, exist_ok=True)

    save_image = image_grid([person_image, cloth_image, result_image], 1, 3)
    save_image.save(result_path)
    print(f"✅ Saved result to: {result_path}")

if __name__ == "__main__":
    main()
