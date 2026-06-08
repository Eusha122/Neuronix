#!/usr/bin/env python3
"""
Download EMNIST-Letters and extract the four IDX files Neuronix needs.

Files written to data/emnist/:
  emnist-letters-train-images-idx3-ubyte
  emnist-letters-train-labels-idx1-ubyte
  emnist-letters-test-images-idx3-ubyte
  emnist-letters-test-labels-idx1-ubyte

Usage:
  python scripts/prepare_emnist.py
  python scripts/prepare_emnist.py --datadir data/emnist
"""

import argparse
import gzip
import shutil
import subprocess
import zipfile
from pathlib import Path

EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

LETTERS_FILES = [
    "emnist-letters-train-images-idx3-ubyte.gz",
    "emnist-letters-train-labels-idx1-ubyte.gz",
    "emnist-letters-test-images-idx3-ubyte.gz",
    "emnist-letters-test-labels-idx1-ubyte.gz",
]

def download(url: str, dest: Path):
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}  (~536 MB, this may take a few minutes) ...")
    result = subprocess.run(
        ["curl", "-L", "--progress-bar", "-o", str(dest), url],
        check=False
    )
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(
            f"curl failed (exit {result.returncode}).\n"
            "Try downloading manually from:\n"
            "  https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip\n"
            "or Kaggle: https://www.kaggle.com/datasets/crawford/emnist\n"
            f"and place gzip.zip in: {dest.parent}"
        )

def extract(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    already = all((out_dir / f.replace(".gz", "")).exists() for f in LETTERS_FILES)
    if already:
        print("Already extracted.")
        return

    print("Extracting letters split from gzip.zip ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip contains a flat list of .gz files
        zip_names = zf.namelist()
        for letters_gz in LETTERS_FILES:
            # find the entry (may be in a subdirectory inside the zip)
            match = next((n for n in zip_names if n.endswith(letters_gz)), None)
            if match is None:
                raise RuntimeError(f"Not found in zip: {letters_gz}")
            gz_dest = out_dir / letters_gz
            print(f"  {match} -> {gz_dest.name}")
            with zf.open(match) as src, open(gz_dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    print("Decompressing .gz files ...")
    for letters_gz in LETTERS_FILES:
        gz_path  = out_dir / letters_gz
        idx_path = out_dir / letters_gz.replace(".gz", "")
        if idx_path.exists():
            gz_path.unlink(missing_ok=True)
            continue
        with gzip.open(gz_path, "rb") as src, open(idx_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz_path.unlink()
        print(f"  {idx_path.name}  ({idx_path.stat().st_size // 1024} KB)")

    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=Path("data/emnist"))
    args = parser.parse_args()

    zip_path = args.datadir / "gzip.zip"
    download(EMNIST_URL, zip_path)
    extract(zip_path, args.datadir)

    # Remove the large zip after extraction
    if zip_path.exists():
        zip_path.unlink()
        print("Removed gzip.zip (no longer needed)")

    print(f"\nEMNIST Letters ready in: {args.datadir}/")
    print("Run: .\\build\\Release\\neuronix_emnist.exe")

if __name__ == "__main__":
    main()
