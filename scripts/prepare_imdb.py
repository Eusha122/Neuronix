#!/usr/bin/env python3
"""
Download the IMDB movie-review dataset and convert it to Neuronix binary format.

Files written to data/imdb/:
  imdb_train.bin   — 25,000 reviews
  imdb_test.bin    — 25,000 reviews
  vocab.txt        — one word per line (line N = token index N)

Binary format per file:
  [4 bytes LE uint32: N]
  [4 bytes LE uint32: seq_len]
  [N * seq_len * 2 bytes: int16 token indices, sample-major]
  [N bytes: uint8 labels — 0=negative, 1=positive]

Usage:
  python scripts/prepare_imdb.py
  python scripts/prepare_imdb.py --seq-len 100 --vocab-size 10000
"""

import argparse
import collections
import re
import struct
import subprocess
import tarfile
from pathlib import Path

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

def download(url: str, dest: Path):
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}  (~84 MB) ...")
    result = subprocess.run(
        ["curl", "-L", "--progress-bar", "-o", str(dest), url],
        check=False
    )
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(
            f"curl failed.\nTry downloading manually from:\n  {url}\n"
            f"and place aclImdb_v1.tar.gz in: {dest.parent}"
        )

def extract(tgz: Path, dest: Path):
    if dest.exists():
        print(f"Already extracted: {dest}")
        return
    print("Extracting ...")
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(dest.parent)
    print("Done.")

def tokenize(text: str) -> list:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def build_vocab(reviews: list, vocab_size: int) -> dict:
    counter = collections.Counter()
    for tokens in reviews:
        counter.update(tokens)
    # Reserve 0 = <PAD>, 1 = <UNK>
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab

def encode(tokens: list, vocab: dict, seq_len: int) -> list:
    unk = vocab["<UNK>"]
    ids = [vocab.get(t, unk) for t in tokens[:seq_len]]
    ids += [0] * (seq_len - len(ids))   # pad with 0
    return ids

def read_split(split_dir: Path) -> tuple:
    reviews, labels = [], []
    for label_idx, sentiment in enumerate(["neg", "pos"]):
        folder = split_dir / sentiment
        for filepath in sorted(folder.glob("*.txt")):
            reviews.append(tokenize(filepath.read_text(encoding="utf-8")))
            labels.append(label_idx)
    return reviews, labels

def write_bin(path: Path, reviews: list, labels: list,
              vocab: dict, seq_len: int):
    N = len(reviews)
    print(f"Writing {N} samples → {path.name} ...")
    with open(path, "wb") as f:
        f.write(struct.pack("<II", N, seq_len))
        for tokens in reviews:
            ids = encode(tokens, vocab, seq_len)
            for idx in ids:
                f.write(struct.pack("<h", idx))   # int16
        for lbl in labels:
            f.write(struct.pack("B", lbl))        # uint8
    mb = path.stat().st_size / 1024 / 1024
    print(f"  Saved {mb:.1f} MB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len",    type=int,  default=100)
    parser.add_argument("--vocab-size", type=int,  default=10000)
    parser.add_argument("--datadir",    type=Path, default=Path("data/imdb"))
    args = parser.parse_args()

    tgz      = args.datadir / "aclImdb_v1.tar.gz"
    imdb_dir = args.datadir / "aclImdb"

    download(IMDB_URL, tgz)
    extract(tgz, imdb_dir)

    print("Reading reviews ...")
    train_reviews, train_labels = read_split(imdb_dir / "train")
    test_reviews,  test_labels  = read_split(imdb_dir / "test")
    print(f"  Train: {len(train_reviews)}  Test: {len(test_reviews)}")

    print(f"Building vocabulary (top {args.vocab_size} words, seq_len={args.seq_len}) ...")
    vocab = build_vocab(train_reviews, args.vocab_size)
    print(f"  Vocab size: {len(vocab)}")

    # Save vocab
    vocab_path = args.datadir / "vocab.txt"
    idx2word = {v: k for k, v in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i in range(len(idx2word)):
            f.write(idx2word[i] + "\n")
    print(f"  Vocab saved to {vocab_path}")

    args.datadir.mkdir(parents=True, exist_ok=True)
    write_bin(args.datadir / "imdb_train.bin",
              train_reviews, train_labels, vocab, args.seq_len)
    write_bin(args.datadir / "imdb_test.bin",
              test_reviews, test_labels, vocab, args.seq_len)

    print(f"\nIMDB ready in: {args.datadir}/")
    print("Run: .\\build\\Release\\neuronix_sentiment.exe")

if __name__ == "__main__":
    main()
