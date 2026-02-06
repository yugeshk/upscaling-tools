#!/bin/bash
# Video/Image upscaling wrapper script using Real-ESRGAN
# Supports both NCNN-Vulkan and PyTorch backends.
# Usage: ./upscale.sh input output [options]

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Dependency checks ───────────────────────────────────────────────
check_dep() {
    if ! command -v "$1" &>/dev/null; then
        echo "Error: $1 is not installed."
        echo "  macOS:  brew install $1"
        echo "  Ubuntu: sudo apt install $1"
        exit 1
    fi
}

check_dep ffmpeg
check_dep python3

# ── Defaults ────────────────────────────────────────────────────────
BACKEND="ncnn"
SCALE=""
MODEL=""
DURATION=""
EXTRA_ARGS=()

# ── Parse arguments ─────────────────────────────────────────────────
INPUT_FILE=""
OUTPUT_FILE=""

show_help() {
    cat <<'HELP'
Usage: ./upscale.sh <input> <output> [options]

Upscale images and videos using Real-ESRGAN.

Options:
  --backend ncnn|pytorch  Backend to use (default: ncnn)
  --scale 2|4             Upscale factor (default: 4)
  --model NAME            Model name (depends on backend)
  --duration SECONDS      Only process first N seconds of video (for testing)
  --help                  Show this help message

Image examples:
  ./upscale.sh photo.jpg photo_4x.jpg
  ./upscale.sh photo.jpg photo_2x.jpg --scale 2
  ./upscale.sh photo.jpg photo_4x.jpg --backend pytorch

Video examples:
  ./upscale.sh video.mp4 upscaled.mp4
  ./upscale.sh video.mp4 upscaled.mp4 --backend pytorch --scale 2
  ./upscale.sh video.mp4 test.mp4 --duration 5

Backends:
  ncnn     Fast GPU upscaling via realesrgan-ncnn-vulkan (default).
           Minimal dependencies — no Python ML libs needed for images.
  pytorch  Full PyTorch pipeline. More tunable (see config.py).
           Needed if you want to adjust tile size, FP16, etc.

Models:
  NCNN:    realesrgan-x4plus (default), realesrgan-x4plus-anime,
           realesr-animevideov3
  PyTorch: RealESRGAN_x4plus (default), RealESRGAN_x4plus_anime,
           RealESRNet_x4plus, realesr-animevideov3

Configuration:
  Edit config.py to change default scale, CRF, tile size, encoding
  preset, and other parameters. See README.md for details.
HELP
    exit 0
}

# Parse positional and named args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            ;;
        --backend)
            BACKEND="$2"; shift 2
            ;;
        --scale)
            SCALE="$2"; shift 2
            ;;
        --model)
            MODEL="$2"; shift 2
            ;;
        --duration)
            DURATION="$2"; shift 2
            ;;
        -*)
            EXTRA_ARGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                EXTRA_ARGS+=("$2"); shift
            fi
            shift
            ;;
        *)
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
            elif [[ -z "$OUTPUT_FILE" ]]; then
                OUTPUT_FILE="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]]; then
    show_help
fi

# ── Build common args ───────────────────────────────────────────────
CMD_ARGS=()
[[ -n "$SCALE" ]] && CMD_ARGS+=(--scale "$SCALE")
[[ -n "$MODEL" ]] && CMD_ARGS+=(--model "$MODEL")
[[ -n "$DURATION" ]] && CMD_ARGS+=(--duration "$DURATION")
CMD_ARGS+=("${EXTRA_ARGS[@]}")

# ── Determine if video or image ─────────────────────────────────────
EXT="${INPUT_FILE##*.}"
EXT_LOWER=$(echo "$EXT" | tr '[:upper:]' '[:lower:]')

if [[ "$EXT_LOWER" == "mp4" || "$EXT_LOWER" == "mov" || "$EXT_LOWER" == "avi" || "$EXT_LOWER" == "mkv" || "$EXT_LOWER" == "webm" ]]; then
    # ── Video ────────────────────────────────────────────────────────
    echo "Processing video file..."
    if [[ "$BACKEND" == "pytorch" ]]; then
        python3 "$SCRIPT_DIR/upscale_video.py" "$INPUT_FILE" "$OUTPUT_FILE" "${CMD_ARGS[@]}"
    else
        python3 "$SCRIPT_DIR/upscale_video_ncnn.py" "$INPUT_FILE" "$OUTPUT_FILE" "${CMD_ARGS[@]}"
    fi
else
    # ── Image ────────────────────────────────────────────────────────
    echo "Processing image file..."
    if [[ "$BACKEND" == "pytorch" ]]; then
        python3 -m realesrgan -i "$INPUT_FILE" -o "$OUTPUT_FILE" "${CMD_ARGS[@]}"
    else
        NCNN_BIN="$SCRIPT_DIR/realesrgan-ncnn-vulkan"
        if [[ ! -f "$NCNN_BIN" ]]; then
            echo "Error: realesrgan-ncnn-vulkan not found."
            echo "Run: bash setup_models.sh"
            exit 1
        fi
        NCNN_ARGS=(-i "$INPUT_FILE" -o "$OUTPUT_FILE")
        [[ -n "$SCALE" ]] && NCNN_ARGS+=(-s "$SCALE")
        [[ -n "$MODEL" ]] && NCNN_ARGS+=(-n "$MODEL")
        NCNN_ARGS+=(-m "$SCRIPT_DIR/models")
        "$NCNN_BIN" "${NCNN_ARGS[@]}"
    fi
fi
