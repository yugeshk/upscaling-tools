#!/bin/bash
# Download Real-ESRGAN model weights and (optionally) the ncnn-vulkan binary.
# Usage:
#   bash setup_models.sh          # download default models only
#   bash setup_models.sh --all    # download all model variants

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/weights"
MODELS_DIR="$SCRIPT_DIR/models"

DOWNLOAD_ALL=false
if [[ "${1:-}" == "--all" ]]; then
    DOWNLOAD_ALL=true
fi

download() {
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  [skip] $(basename "$dest") already exists"
        return
    fi
    echo "  [download] $(basename "$dest")"
    curl -L --progress-bar -o "$dest" "$url"
}

# ── PyTorch weights ─────────────────────────────────────────────────
mkdir -p "$WEIGHTS_DIR"
echo "=== PyTorch model weights ==="

# Default model (always downloaded)
download "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
         "$WEIGHTS_DIR/RealESRGAN_x4plus.pth"

if $DOWNLOAD_ALL; then
    download "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" \
             "$WEIGHTS_DIR/RealESRGAN_x4plus_anime_6B.pth"
    download "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth" \
             "$WEIGHTS_DIR/RealESRNet_x4plus.pth"
    download "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth" \
             "$WEIGHTS_DIR/realesr-animevideov3.pth"
fi

# ── NCNN models (.bin + .param) ─────────────────────────────────────
mkdir -p "$MODELS_DIR"
echo ""
echo "=== NCNN model files ==="

NCNN_RELEASE="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0"

# Default model (always downloaded)
for ext in bin param; do
    download "$NCNN_RELEASE/realesrgan-x4plus.$ext" "$MODELS_DIR/realesrgan-x4plus.$ext"
done

if $DOWNLOAD_ALL; then
    for model in realesrgan-x4plus-anime realesr-animevideov3; do
        for ext in bin param; do
            download "$NCNN_RELEASE/$model.$ext" "$MODELS_DIR/$model.$ext"
        done
    done
fi

# ── NCNN binary ─────────────────────────────────────────────────────
echo ""
echo "=== realesrgan-ncnn-vulkan binary ==="

NCNN_BIN="$SCRIPT_DIR/realesrgan-ncnn-vulkan"
if [[ -f "$NCNN_BIN" ]]; then
    echo "  [skip] realesrgan-ncnn-vulkan already exists"
else
    NCNN_BIN_RELEASE="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0"
    OS="$(uname -s)"
    case "$OS" in
        Darwin)
            ZIP_NAME="realesrgan-ncnn-vulkan-20220424-macos.zip"
            ;;
        Linux)
            ZIP_NAME="realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
            ;;
        *)
            echo "  [error] Unsupported OS: $OS. Download manually from:"
            echo "    https://github.com/xinntao/Real-ESRGAN/releases"
            exit 1
            ;;
    esac

    echo "  [download] $ZIP_NAME"
    TMPDIR="$(mktemp -d)"
    curl -L --progress-bar -o "$TMPDIR/$ZIP_NAME" "$NCNN_BIN_RELEASE/$ZIP_NAME"
    unzip -q "$TMPDIR/$ZIP_NAME" -d "$TMPDIR"
    cp "$TMPDIR/realesrgan-ncnn-vulkan" "$NCNN_BIN"
    chmod +x "$NCNN_BIN"
    rm -rf "$TMPDIR"
fi

echo ""
echo "=== Setup complete ==="
echo "Models directory: $MODELS_DIR"
echo "Weights directory: $WEIGHTS_DIR"
