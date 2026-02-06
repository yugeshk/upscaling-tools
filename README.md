# upscaling-tools

Image and video upscaling toolkit built on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), with two backends: **NCNN-Vulkan** (fast, minimal deps) and **PyTorch** (flexible, tunable).

## Quick Start

```bash
git clone https://github.com/yugeshk/upscaling-tools.git
cd upscaling-tools
bash setup_models.sh
./upscale.sh photo.jpg photo_4x.jpg
```

## Prerequisites

- **ffmpeg** (required for video processing)
- **Python 3.8+** (required for PyTorch backend and video scripts)
- A Vulkan-capable GPU (for NCNN backend)

## Installation

### NCNN path (minimal — no Python ML deps needed for images)

```bash
bash setup_models.sh
```

This downloads the `realesrgan-ncnn-vulkan` binary and default model files. You can upscale images immediately:

```bash
./upscale.sh image.jpg output.jpg
```

### PyTorch path (full ML stack — needed for video upscaling via PyTorch)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash setup_models.sh
```

## Usage

### Upscale an image

```bash
# NCNN backend (default, fast)
./upscale.sh photo.jpg photo_4x.jpg

# PyTorch backend
./upscale.sh photo.jpg photo_4x.jpg --backend pytorch

# 2x upscale instead of 4x
./upscale.sh photo.jpg photo_2x.jpg --scale 2
```

### Upscale a video

```bash
# Auto-detects video by extension, uses NCNN backend by default
./upscale.sh video.mp4 video_upscaled.mp4

# PyTorch backend
./upscale.sh video.mp4 video_upscaled.mp4 --backend pytorch

# Test on first 5 seconds
./upscale.sh video.mp4 test_output.mp4 --duration 5
```

### Direct Python script usage

```bash
# PyTorch backend with custom tile size
python3 upscale_video.py input.mp4 output.mp4 --tile-size 256 --scale 2

# NCNN backend with duration limit
python3 upscale_video_ncnn.py input.mp4 output.mp4 --scale 2 --duration 10

# Keep extracted frames for inspection
python3 upscale_video.py input.mp4 output.mp4 --keep-frames
```

## Tuning / Configuration

All tunable parameters live in **`config.py`**. Edit this file to change defaults across all scripts.

Key parameters:

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `TILE_SIZE` | 512 | Tile size for PyTorch processing. **Reduce to 256 or 128 if you get OOM errors.** |
| `USE_HALF` | False | FP16 mode. Set `True` on NVIDIA GPUs for ~2x speed. |
| `DEFAULT_CRF` | 18 | Video quality. 0 = lossless, 18 = visually lossless, 28 = low quality. |
| `VIDEO_PRESET` | slow | Encoding speed/quality. `ultrafast` for testing, `slow` for final output. |
| `AUDIO_BITRATE` | 192k | Audio bitrate. 128k for speech, 256-320k for music. |

## Available Models

| Model | Best for |
|-------|----------|
| `RealESRGAN_x4plus` | General-purpose photos (default) |
| `RealESRGAN_x4plus_anime` | Anime / illustration images |
| `RealESRNet_x4plus` | Photos where you want sharper, less smoothed output |
| `realesr-animevideov3` | Anime video frames, faster processing |

Download all models with `bash setup_models.sh --all`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of memory (OOM)** | Reduce `TILE_SIZE` in `config.py` to 256 or 128 |
| **Slow processing** | Use NCNN backend (`--backend ncnn`), or set `USE_HALF=True` for NVIDIA GPUs |
| **Binary not found** | Run `bash setup_models.sh` to download the ncnn-vulkan binary |
| **ffmpeg not found** | Install ffmpeg: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Ubuntu) |
| **Poor output quality** | Try a different model, or lower the CRF value in `config.py` |
