"""
Centralized configuration for upscaling-tools.
Edit values here to tune upscaling behavior across all scripts.
"""

# ── Upscaling defaults ──────────────────────────────────────────────
DEFAULT_SCALE = 4                          # 2 or 4; higher = more detail but slower
DEFAULT_MODEL_PYTORCH = "RealESRGAN_x4plus"  # best general-purpose model
DEFAULT_MODEL_NCNN = "realesrgan-x4plus"     # ncnn model name (no .pth)

# ── Tile processing (PyTorch backend) ───────────────────────────────
TILE_SIZE = 512        # pixels; reduce to 256 or 128 if you get OOM errors
TILE_PAD = 10          # overlap between tiles to avoid seam artifacts
USE_HALF = False       # True = FP16, faster on NVIDIA GPUs; False = FP32 for compatibility

# ── Video encoding (ffmpeg) ─────────────────────────────────────────
VIDEO_CODEC = "libx264"    # libx264 (CPU), libx265 (smaller files), h264_videotoolbox (macOS HW)
VIDEO_PRESET = "slow"      # ultrafast → fast → medium → slow → veryslow; slower = smaller file
DEFAULT_CRF = 18           # 0 = lossless, 18 = visually lossless, 23 = default, 28 = low quality
AUDIO_BITRATE = "192k"     # audio re-encode bitrate; 128k for speech, 192-320k for music
PIX_FMT = "yuv420p"        # pixel format; yuv420p for max compatibility

# ── PyTorch model configurations ────────────────────────────────────
# Each model maps to its architecture params, weight filename, and download URL.
PYTORCH_MODELS = {
    "RealESRGAN_x4plus": {
        "num_block": 23,
        "num_grow_ch": 32,
        "netscale": 4,
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "description": "Best general-purpose photo upscaler",
    },
    "RealESRGAN_x4plus_anime": {
        "num_block": 6,
        "num_grow_ch": 32,
        "netscale": 4,
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "description": "Optimized for anime/illustration images",
    },
    "RealESRNet_x4plus": {
        "num_block": 23,
        "num_grow_ch": 32,
        "netscale": 4,
        "filename": "RealESRNet_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "description": "Sharper output, less smoothing than x4plus",
    },
    "realesr-animevideov3": {
        "num_block": 6,
        "num_grow_ch": 32,
        "netscale": 4,
        "filename": "realesr-animevideov3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "description": "Designed for anime video frames, fast",
    },
}

# ── NCNN model names (used by realesrgan-ncnn-vulkan) ───────────────
NCNN_MODELS = [
    "realesrgan-x4plus",
    "realesrgan-x4plus-anime",
    "realesr-animevideov3",
]
