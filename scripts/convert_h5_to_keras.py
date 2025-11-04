#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Convert TF-Keras .h5 models to Keras (.keras) format.")
    p.add_argument("paths", nargs="+", help="Files or directories to convert.")
    p.add_argument("--recursive", "-r", action="store_true", help="Recurse into directories.")
    p.add_argument("--out", "-o", default=None, help="Output directory (mirrors structure). Default: alongside source.")
    args = p.parse_args()

    try:
        import tensorflow as tf  # only needed for conversion
    except Exception:
        print("TensorFlow is required only for conversion. Install temporarily:", file=sys.stderr)
        print("  pip install 'tensorflow>=2.9'  # or tensorflow-macos/tensorflow-metal on Apple Silicon", file=sys.stderr)
        return 2

    # Collect .h5 files
    files = []
    for pth in args.paths:
        p = Path(pth)
        if p.is_file() and p.suffix.lower() == ".h5":
            files.append(p)
        elif p.is_dir():
            it = p.rglob("*.h5") if args.recursive else p.glob("*.h5")
            files.extend(list(it))
        else:
            print(f"Skipping non-existent path: {p}", file=sys.stderr)

    if not files:
        print("No .h5 files found.", file=sys.stderr)
        return 0

    for src in files:
        try:
            model = tf.keras.models.load_model(str(src))
        except Exception as e:
            print(f"Failed to load {src}: {e}", file=sys.stderr)
            continue

        if args.out:
            dst_dir = Path(args.out)
            dst = dst_dir / src.with_suffix(".keras").name
        else:
            dst_dir = src.parent
            dst = src.with_suffix(".keras")

        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save(str(dst))
            print(f"Converted: {src} -> {dst}")
        except Exception as e:
            print(f"Failed to save {src} -> {dst}: {e}", file=sys.stderr)

    return 0

if __name__ == "__main__":
    sys.exit(main())
