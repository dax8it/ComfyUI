import os
import hashlib
import subprocess
import tempfile
import random
from PIL import Image
import torch
import numpy as np

MAX_SEED = 0x7fffffffffffffff  # 9223372036854775807 (int64 max)

try:
    import folder_paths  # ComfyUI helper
except Exception:
    folder_paths = None

def _autofix_flux_paths(flux_bin: str, model_dir: str) -> tuple[str, str]:
    fb = os.path.expanduser(flux_bin or "")
    md = os.path.expanduser(model_dir or "")

    # If flux_bin is accidentally set to the model directory, detect it by structure.
    if fb and os.path.isdir(fb):
        looks_like_model = any(
            os.path.isdir(os.path.join(fb, d)) for d in ("transformer", "vae", "text_encoder", "tokenizer")
        )
        if looks_like_model:
            # Treat this directory as model_dir
            md = fb
            # Try to locate the flux executable next to it (flux2.c repo layout)
            cand = os.path.join(os.path.dirname(fb), "flux")
            fb = cand

    return fb, md

def _validate_flux_bin(flux_bin_path):
    if not flux_bin_path:
        raise RuntimeError("flux_bin is not set")
    if not os.path.exists(flux_bin_path):
        raise RuntimeError(f"flux_bin '{flux_bin_path}' does not exist")
    if os.path.isdir(flux_bin_path):
        raise RuntimeError(f"flux_bin '{flux_bin_path}' is a directory, not an executable")
    if not os.access(flux_bin_path, os.X_OK):
        raise RuntimeError(f"flux_bin '{flux_bin_path}' is not executable")

def ensure_flux_binary(flux_bin_path):
    try:
        _validate_flux_bin(flux_bin_path)
        return True
    except RuntimeError as e:
        msg = str(e)
        # If flux_bin is a directory, do NOT try building; this is a user/config mistake.
        if "is a directory" in msg:
            raise RuntimeError(
                f"{msg}\n\n"
                "It looks like flux_bin was set to the model folder.\n"
                "Fix: set flux_bin to /Users/alexcovo/Documents/GITHUB/flux2.c/flux\n"
                "and set model_dir to /Users/alexcovo/Documents/GITHUB/flux2.c/flux-klein-model\n"
            )

        # Otherwise, try to build as a fallback.
        repo_path = "/Users/alexcovo/Documents/GITHUB/flux2.c"
        if not os.path.exists(repo_path):
            raise RuntimeError(f"Flux2.c repo not found at {repo_path}")
        try:
            result = subprocess.run(["make", "mps"], cwd=repo_path, capture_output=True, text=True)
            if result.returncode == 0:
                _validate_flux_bin(flux_bin_path)
                return True
            raise RuntimeError(f"Failed to build flux binary. Make output: {result.stdout + result.stderr}")
        except Exception as e2:
            raise RuntimeError(f"Error building flux binary: {str(e2)}")

def _get_comfy_root() -> str:
    # nodes.py is at: <ComfyUI>/custom_nodes/comfyui_flux2c/nodes.py
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _get_default_output_dir() -> str:
    """
    Prefer ComfyUI output dir if available; otherwise use the user's requested path:
    <ComfyUI>/user/output/flux_c
    """
    sub = "flux_c"

    # 1) ComfyUI-configured output directory
    if folder_paths is not None:
        try:
            base = folder_paths.get_output_directory()
            out = os.path.join(base, sub)
            os.makedirs(out, exist_ok=True)
            return out
        except Exception:
            pass

    # 2) User requested: <ComfyUI>/user/output/flux_c
    root = _get_comfy_root()
    base = os.path.join(root, "user", "output")
    os.makedirs(base, exist_ok=True)
    out = os.path.join(base, sub)
    os.makedirs(out, exist_ok=True)
    return out

def _make_output_path(prefix: str, seed: int, prompt: str, ext=".png") -> str:
    out_dir = _get_default_output_dir()
    h = hashlib.sha1(prompt.encode("utf-8", errors="ignore")).hexdigest()[:10]
    filename = f"{prefix}_{seed}_{h}{ext}"
    return os.path.join(out_dir, filename)

def _sanitize_prompt(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw)

    # Remove null bytes
    s = s.replace("\x00", "")

    # If a ```json fenced block exists, try extracting prompt from JSON
    import re, json
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            data = json.loads(m.group(1))
            p = data.get("prompt") or data.get("positive_prompt") or data.get("text") or ""
            if isinstance(p, str) and p.strip():
                s = p
        except Exception:
            # fall back to cleaning below
            pass

    # Remove any remaining fenced blocks completely
    s = re.sub(r"```.*?```", "", s, flags=re.DOTALL)

    # Strip common assistant preface lines
    # If the user pasted a whole assistant response, keep only the last "real" paragraph
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if len(lines) > 0:
        # If it starts with "I'll" or similar, drop the first line
        if re.match(r"^(i('| a)m|i will|sure|here('| a)re|okay|ok)\b", lines[0].lower()):
            lines = lines[1:]
    s = " ".join(lines).strip()

    # Hard length cap (avoid extreme prompts)
    if len(s) > 2000:
        s = s[:2000].rstrip()

    return s

class Flux2CTxt2Img:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flux_bin": ("STRING", {"default": "/Users/alexcovo/Documents/GITHUB/flux2.c/flux"}),
                "model_dir": ("STRING", {"default": "/Users/alexcovo/Documents/GITHUB/flux2.c/flux-klein-model"}),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "width": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 16}),
                "height": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 16}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7fffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Flux2C"

    def generate(self, flux_bin, model_dir, prompt, width, height, steps, guidance, seed):
        flux_bin, model_dir = _autofix_flux_paths(flux_bin, model_dir)
        ensure_flux_binary(flux_bin)

        if seed == -1:
            seed = random.randrange(0, 0x7fffffffffffffff)
        seed = int(seed)
        if seed < 0:
            seed = 0
        if seed > 0x7fffffffffffffff:
            seed = seed % 0x7fffffffffffffff

        prompt = _sanitize_prompt(prompt)
        if not prompt:
            raise RuntimeError("Prompt is empty after sanitization. Please provide a plain text prompt.")

        # Resolve model_dir if relative
        resolved_model_dir = os.path.abspath(model_dir)

        out_path = _make_output_path("flux2c_txt2img", seed, prompt, ".png")

        cmd = [
            flux_bin,
            "-d", resolved_model_dir,
            "-p", prompt,
            "-o", out_path,
            "-W", str(width),
            "-H", str(height),
            "-s", str(steps),
            "-g", str(guidance),
            "-S", str(seed),
        ]

        result = subprocess.run(cmd, cwd=os.path.dirname(flux_bin), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Flux txt2img generation failed.\n\n"
                f"Resolved model_dir: {resolved_model_dir}\n"
                f"Output path: {out_path}\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"stderr:\n{result.stderr}"
            )

        img = Image.open(out_path).convert("RGB")
        # Convert to tensor for ComfyUI
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        return (img_tensor.unsqueeze(0),)

class Flux2CImg2Img:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flux_bin": ("STRING", {"default": "/Users/alexcovo/Documents/GITHUB/flux2.c/flux"}),
                "model_dir": ("STRING", {"default": "/Users/alexcovo/Documents/GITHUB/flux2.c/flux-klein-model"}),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape"}),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7fffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Flux2C"

    def generate(self, flux_bin, model_dir, prompt, image, strength, steps, guidance, seed):
        flux_bin, model_dir = _autofix_flux_paths(flux_bin, model_dir)
        ensure_flux_binary(flux_bin)

        if seed == -1:
            seed = random.randrange(0, 0x7fffffffffffffff)
        seed = int(seed)
        if seed < 0:
            seed = 0
        if seed > 0x7fffffffffffffff:
            seed = seed % 0x7fffffffffffffff

        prompt = _sanitize_prompt(prompt)
        if not prompt:
            raise RuntimeError("Prompt is empty after sanitization. Please provide a plain text prompt.")

        # Resolve model_dir if relative
        resolved_model_dir = os.path.abspath(model_dir)

        # Save input image to temp file
        img = Image.fromarray((image.squeeze(0).numpy() * 255).astype(np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            input_path = f.name
            img.save(input_path)

        out_path = _make_output_path("flux2c_img2img", seed, prompt, ".png")

        cmd = [
            flux_bin,
            "-d", resolved_model_dir,
            "-p", prompt,
            "-i", input_path,
            "-o", out_path,
            "-t", str(strength),
            "-s", str(steps),
            "-g", str(guidance),
            "-S", str(seed),
        ]

        result = subprocess.run(cmd, cwd=os.path.dirname(flux_bin), capture_output=True, text=True)
        os.unlink(input_path)  # Clean up temp input
        if result.returncode != 0:
            raise RuntimeError(
                "Flux img2img generation failed.\n\n"
                f"Resolved model_dir: {resolved_model_dir}\n"
                f"Output path: {out_path}\n"
                f"Command: {' '.join(cmd)}\n\n"
                f"stderr:\n{result.stderr}"
            )

        img = Image.open(out_path).convert("RGB")
        # Convert to tensor for ComfyUI
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        return (img_tensor.unsqueeze(0),)

NODE_CLASS_MAPPINGS = {
    "Flux2CTxt2Img": Flux2CTxt2Img,
    "Flux2CImg2Img": Flux2CImg2Img,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2CTxt2Img": "Flux2C Txt2Img",
    "Flux2CImg2Img": "Flux2C Img2Img",
}
