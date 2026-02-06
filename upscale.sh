#!/bin/bash
# Video/Image upscaling wrapper script using Real-ESRGAN
# Usage: ./upscale.sh input.mp4 output.mp4 [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

# Activate virtualenv
source "$VENV_DIR/bin/activate"

# Check if it's a video or image based on extension
INPUT_FILE="$1"
OUTPUT_FILE="$2"
shift 2

if [[ -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]]; then
    echo "Usage: ./upscale.sh <input> <output> [options]"
    echo ""
    echo "For video:"
    echo "  ./upscale.sh video.mp4 video_upscaled.mp4 --scale 2"
    echo ""
    echo "For images:"
    echo "  ./upscale.sh image.jpg image_upscaled.jpg --scale 4"
    echo ""
    echo "Options:"
    echo "  --scale 2|4    Upscale factor (default: 4)"
    echo "  --model NAME   Model to use (default: RealESRGAN_x4plus)"
    echo ""
    exit 1
fi

# Determine if video or image
EXT="${INPUT_FILE##*.}"
EXT_LOWER=$(echo "$EXT" | tr '[:upper:]' '[:lower:]')

if [[ "$EXT_LOWER" == "mp4" || "$EXT_LOWER" == "mov" || "$EXT_LOWER" == "avi" || "$EXT_LOWER" == "mkv" || "$EXT_LOWER" == "webm" ]]; then
    echo "Processing video file..."
    python3 "$SCRIPT_DIR/upscale_video.py" "$INPUT_FILE" "$OUTPUT_FILE" "$@"
else
    echo "Processing image file..."
    # For images, use the realesrgan command directly
    python3 -m realesrgan -i "$INPUT_FILE" -o "$OUTPUT_FILE" "$@"
fi

deactivate
