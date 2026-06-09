#!/usr/bin/env python3
"""
Run inference on a single image using a trained Neuronix model.

Usage:
  python scripts/infer.py photo.jpg --model emnist_letters.nnx --task emnist
  python scripts/infer.py face.jpg  --model face_classifier.nnx --task face
  python scripts/infer.py img.jpg   --model cifar10_vgg.nnx --task cifar10
  python scripts/infer.py digit.jpg --model lenet_mnist.nnx --task mnist

Options:
  --top N     Show top N predictions (default: 3)
  --exe PATH  Path to neuronix_infer.exe (default: build/Release/neuronix_infer.exe)
"""

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

# Preprocessing config per task — must match training exactly
TASKS = {
    'emnist': {
        'size':     (28, 28),
        'channels': 1,
        'mean':     [0.1722],
        'std':      [0.3309],
        'labels':   [chr(ord('A') + i) for i in range(26)],
    },
    'mnist': {
        'size':     (28, 28),
        'channels': 1,
        'mean':     [0.0],       # MNIST loader only divides by 255, no extra norm
        'std':      [1.0],
        'labels':   [str(i) for i in range(10)],
    },
    'cifar10': {
        'size':     (32, 32),
        'channels': 3,
        'mean':     [0.0, 0.0, 0.0],   # CIFAR loader only divides by 255
        'std':      [1.0, 1.0, 1.0],
        'labels':   ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck'],
    },
    'face': {
        'size':     (32, 32),
        'channels': 3,
        'mean':     [0.0, 0.0, 0.0],   # face loader only divides by 255
        'std':      [1.0, 1.0, 1.0],
        'labels':   ['no-face', 'face'],
    },
}

def check_pil():
    try:
        from PIL import Image
        return True
    except ImportError:
        print("Missing Pillow. Run:  pip install Pillow")
        return False

def preprocess(image_path: Path, cfg: dict) -> list:
    from PIL import Image
    import numpy as np

    H, W   = cfg['size']
    C      = cfg['channels']
    mode   = 'L' if C == 1 else 'RGB'

    img = Image.open(image_path).convert(mode).resize((W, H), Image.BILINEAR)
    arr = np.array(img, dtype=np.float64) / 255.0

    if C == 1:
        arr = arr[np.newaxis, :, :]          # (1, H, W)
    else:
        arr = arr.transpose(2, 0, 1)         # (C, H, W) channel-first

    # Normalise per channel
    for c in range(C):
        arr[c] = (arr[c] - cfg['mean'][c]) / cfg['std'][c]

    return arr.flatten().tolist()

def write_pixel_bin(pixels: list, path: str):
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(pixels)))
        for v in pixels:
            f.write(struct.pack('<d', v))

def main():
    parser = argparse.ArgumentParser(description='Neuronix single-image inference')
    parser.add_argument('image',           type=Path,  help='Input image (PNG/JPG/BMP)')
    parser.add_argument('--model',  '-m',  type=Path,  required=True,
                        help='Path to .nnx model file')
    parser.add_argument('--task',   '-t',  choices=list(TASKS.keys()), required=True,
                        help='Task type: emnist / mnist / cifar10 / face')
    parser.add_argument('--top',    '-k',  type=int,   default=3,
                        help='Number of top predictions to show (default: 3)')
    parser.add_argument('--exe',           type=Path,
                        default=Path('build/Release/neuronix_infer.exe'),
                        help='Path to neuronix_infer.exe')
    args = parser.parse_args()

    if not check_pil():
        sys.exit(1)

    if not args.image.exists():
        print(f"Image not found: {args.image}"); sys.exit(1)
    if not args.model.exists():
        print(f"Model not found: {args.model}"); sys.exit(1)
    if not args.exe.exists():
        print(f"Inference binary not found: {args.exe}")
        print("Build it with:  cmake --build build --config Release --target neuronix_infer")
        sys.exit(1)

    cfg    = TASKS[args.task]
    pixels = preprocess(args.image, cfg)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        tmp_path = tmp.name
    write_pixel_bin(pixels, tmp_path)

    try:
        result = subprocess.run(
            [str(args.exe), str(args.model), tmp_path],
            capture_output=True, text=True
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print("Inference error:", result.stderr.strip())
        sys.exit(1)

    lines       = result.stdout.strip().split('\n')
    num_classes = int(lines[0])
    probs       = [float(lines[i + 1]) for i in range(num_classes)]
    labels      = cfg['labels']

    ranked = sorted(range(num_classes), key=lambda i: probs[i], reverse=True)

    print(f"\n  Image : {args.image.name}")
    print(f"  Model : {args.model.name}")
    print(f"  Task  : {args.task.upper()}\n")
    print(f"  {'Rank':<5} {'Label':<14} {'Confidence':>10}")
    print(f"  {'-'*34}")
    for rank, idx in enumerate(ranked[:args.top], 1):
        label = labels[idx] if idx < len(labels) else str(idx)
        bar   = '█' * int(probs[idx] * 20)
        print(f"  #{rank:<4} {label:<14} {probs[idx]*100:>9.2f}%  {bar}")
    print()

if __name__ == '__main__':
    main()
