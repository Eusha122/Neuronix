#!/usr/bin/env python3
"""
Download LFW-funneled faces and convert to Neuronix binary format.

Format written to lfw_faces_32.bin:
  [4 bytes little-endian uint32: N]
  [N * 3072 bytes: each image = 1024 R + 1024 G + 1024 B, uint8 channel-first]

Usage:
  python scripts/prepare_lfw.py
  python scripts/prepare_lfw.py --size 32 --out data/lfw_faces_32.bin
"""

import argparse
import os
import struct
import subprocess
import tarfile
from pathlib import Path

def check_deps():
    try:
        from PIL import Image
        import numpy as np
        return True
    except ImportError:
        print("Missing dependencies. Run:")
        print("  pip install Pillow numpy")
        return False

def download(url: str, dest: Path):
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    # Use curl (available on Windows 10/11) — avoids Python urllib proxy/DNS issues
    result = subprocess.run(
        ["curl", "-L", "--progress-bar", "-o", str(dest), url],
        check=False
    )
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(f"curl failed to download {url}")

def extract(tgz: Path, dest: Path):
    if dest.exists():
        print(f"Already extracted: {dest}")
        return
    print(f"Extracting {tgz} ...")
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(dest.parent)
    print("Done.")

def convert(face_dir: Path, out_file: Path, size: int):
    from PIL import Image
    import numpy as np

    jpg_files = sorted(face_dir.rglob("*.jpg"))
    N = len(jpg_files)
    if N == 0:
        raise RuntimeError(f"No .jpg files found in {face_dir}")
    print(f"Converting {N} images to {size}x{size} ...")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(struct.pack("<I", N))
        for i, path in enumerate(jpg_files):
            img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1).flatten()
            f.write(arr.tobytes())
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{N}")

    mb = out_file.stat().st_size / 1024 / 1024
    print(f"Saved {N} images → {out_file}  ({mb:.1f} MB)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",    type=int,  default=32,
                        help="Output image size (default: 32)")
    parser.add_argument("--out",     type=Path,
                        default=Path("data/lfw_faces_32.bin"),
                        help="Output binary file path")
    parser.add_argument("--datadir", type=Path, default=Path("data"),
                        help="Directory for downloads/extraction")
    args = parser.parse_args()

    if not check_deps():
        return

    tgz      = args.datadir / "lfw-funneled.tgz"
    face_dir = args.datadir / "lfw_funneled"

    download("https://ndownloader.figshare.com/files/14329295", tgz)
    extract(tgz, face_dir)
    convert(face_dir, args.out, args.size)

if __name__ == "__main__":
    main()
