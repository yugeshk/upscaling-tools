#!/usr/bin/env python3
"""
Video upscaling using Real-ESRGAN ncnn-vulkan (GPU accelerated)
Usage: python upscale_video_ncnn.py input.mp4 output.mp4 [--scale 2|4] [--model MODEL]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from tqdm import tqdm

import config

def run_cmd(cmd, desc=None, capture=True):
    """Run a shell command and print output"""
    if desc:
        print(f"\n>>> {desc}")
    print(f"    {' '.join(cmd)}")
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result
    else:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(1)
        return result

def get_video_info(input_path):
    """Get video fps and resolution using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'csv=p=0',
        input_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    parts = result.stdout.strip().split(',')
    width, height = int(parts[0]), int(parts[1])
    fps_parts = parts[2].split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    return width, height, fps

def main():
    parser = argparse.ArgumentParser(description='Upscale video using Real-ESRGAN ncnn-vulkan (GPU)')
    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output video file')
    parser.add_argument('--scale', type=int, choices=[2, 3, 4], default=config.DEFAULT_SCALE,
                        help=f'Upscale factor (default: {config.DEFAULT_SCALE})')
    parser.add_argument('--model', default=config.DEFAULT_MODEL_NCNN, choices=config.NCNN_MODELS,
                        help=f'Model to use (default: {config.DEFAULT_MODEL_NCNN})')
    parser.add_argument('--fps', type=float, help='Output FPS (default: same as input)')
    parser.add_argument('--keep-frames', action='store_true', help='Keep extracted frames after processing')
    parser.add_argument('--crf', type=int, default=config.DEFAULT_CRF,
                        help=f'Output video CRF quality (default: {config.DEFAULT_CRF})')
    parser.add_argument('--duration', type=float, help='Only process first N seconds (for testing)')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    script_dir = Path(__file__).parent.resolve()
    ncnn_binary = script_dir / 'realesrgan-ncnn-vulkan'

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if not ncnn_binary.exists():
        print(f"Error: ncnn binary not found: {ncnn_binary}")
        print("Run: bash setup_models.sh")
        sys.exit(1)

    # Get video info
    print(f"\n=== Video Upscaling with Real-ESRGAN ncnn-vulkan (GPU) ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    width, height, fps = get_video_info(str(input_path))
    output_fps = args.fps or fps
    print(f"Input resolution: {width}x{height}")
    print(f"Input FPS: {fps}")
    print(f"Scale factor: {args.scale}x")
    print(f"Output resolution: {width * args.scale}x{height * args.scale}")
    print(f"Model: {args.model}")

    # Create temp directories
    temp_dir = Path(tempfile.mkdtemp(prefix='upscale_ncnn_'))
    input_frames_dir = temp_dir / 'input'
    output_frames_dir = temp_dir / 'output'

    input_frames_dir.mkdir(parents=True)
    output_frames_dir.mkdir(parents=True)

    try:
        # Step 1: Extract frames
        extract_cmd = [
            'ffmpeg', '-y', '-i', str(input_path),
            '-qscale:v', '1', '-qmin', '1', '-qmax', '1',
        ]
        if args.duration:
            extract_cmd.extend(['-t', str(args.duration)])
        extract_cmd.append(str(input_frames_dir / 'frame_%08d.png'))

        run_cmd(extract_cmd, "Extracting frames from video...")

        # Count frames
        frame_files = sorted(input_frames_dir.glob('*.png'))
        frame_count = len(frame_files)
        print(f"    Extracted {frame_count} frames")

        # Step 2: Upscale frames with Real-ESRGAN ncnn-vulkan
        print(f"\n>>> Upscaling frames with Real-ESRGAN ncnn-vulkan ({args.model})...")
        print(f"    Processing {frame_count} frames...")

        # Process each frame with progress bar
        for frame_file in tqdm(frame_files, desc="Upscaling"):
            output_file = output_frames_dir / frame_file.name
            cmd = [
                str(ncnn_binary),
                '-i', str(frame_file),
                '-o', str(output_file),
                '-s', str(args.scale),
                '-n', args.model,
                '-m', str(script_dir / 'models')
            ]
            subprocess.run(cmd, capture_output=True)

        # Step 3: Reassemble video
        run_cmd([
            'ffmpeg', '-y',
            '-framerate', str(output_fps),
            '-i', str(output_frames_dir / 'frame_%08d.png'),
            '-c:v', config.VIDEO_CODEC,
            '-crf', str(args.crf),
            '-preset', config.VIDEO_PRESET,
            '-pix_fmt', config.PIX_FMT,
            '-movflags', '+faststart',
            str(output_path)
        ], "Reassembling video...")

        # Copy audio if present
        print("\n>>> Checking for audio track...")
        audio_check = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', str(input_path)],
            capture_output=True, text=True
        )
        if audio_check.stdout.strip():
            print("    Audio found, adding to output...")
            temp_output = output_path.with_suffix('.temp.mp4')
            shutil.move(str(output_path), str(temp_output))
            run_cmd([
                'ffmpeg', '-y',
                '-i', str(temp_output),
                '-i', str(input_path),
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', config.AUDIO_BITRATE,
                '-map', '0:v:0', '-map', '1:a:0?',
                '-shortest',
                str(output_path)
            ], "Adding audio track...")
            temp_output.unlink()
        else:
            print("    No audio track found")

        print(f"\n=== Done! ===")
        print(f"Output saved to: {output_path}")

    finally:
        # Cleanup
        if not args.keep_frames and temp_dir.exists():
            print("\n>>> Cleaning up temporary files...")
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
