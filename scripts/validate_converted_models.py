#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def pair_from_dir(models_dir: Path):
    pairs = []
    for child in sorted(models_dir.iterdir()):
        if child.is_dir() and child.name.endswith("_savedmodel"):
            keras_path = child.with_suffix("")
            keras_path = child.parent / (child.name + ".keras")
            if keras_path.exists():
                pairs.append((child, keras_path))
            else:
                print(f"Skip (no .keras peer): {child}", file=sys.stderr)
    return pairs


def run_savedmodel(path: Path, x):
    import keras
    layer = keras.layers.TFSMLayer(str(path), call_endpoint='serving_default')
    y = layer(x)
    if isinstance(y, dict):
        # Take the first output if dict
        y = list(y.values())[0]
    return y


def run_keras(path: Path, x):
    import keras
    m = keras.models.load_model(str(path), compile=False)
    return m(x, training=False)


def main():
    p = argparse.ArgumentParser(description="Validate converted .keras models against their TF SavedModel counterparts on random inputs.")
    p.add_argument("--savedmodel", type=str, default=None, help="Path to a single SavedModel directory to test.")
    p.add_argument("--keras", type=str, default=None, help="Path to the corresponding .keras model to test.")
    p.add_argument("--dir", type=str, default=None, help="Directory containing *_savedmodel folders and .keras files.")
    p.add_argument("--cadences", type=int, required=True, help="Input length (time steps) expected by the models.")
    p.add_argument("--num", type=int, default=64, help="Batch size of random samples to compare.")
    p.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for equality check.")
    args = p.parse_args()

    try:
        import numpy as np
        import keras
    except Exception as e:
        print("Please install keras>=3 and its TensorFlow backend to run this validation.", file=sys.stderr)
        print("  pip install 'keras>=3' 'tensorflow-macos>=2.15' tensorflow-metal", file=sys.stderr)
        return 2

    pairs = []
    if args.dir:
        pairs.extend(pair_from_dir(Path(os.path.expanduser(args.dir))))
    elif args.savedmodel and args.keras:
        pairs.append((Path(os.path.expanduser(args.savedmodel)), Path(os.path.expanduser(args.keras))))
    else:
        print("Provide either --dir or both --savedmodel and --keras.", file=sys.stderr)
        return 2

    if not pairs:
        print("No model pairs found.", file=sys.stderr)
        return 1

    # Generate a reproducible random batch
    rng = np.random.default_rng(123)
    x = rng.normal(size=(args.num, args.cadences, 1)).astype("float32")

    all_ok = True
    for sm_path, k_path in pairs:
        print(f"Comparing:\n  SavedModel: {sm_path}\n  Keras:      {k_path}")
        y_sm = run_savedmodel(sm_path, x)
        y_k = run_keras(k_path, x)
        # Ensure numpy arrays
        y_sm = np.array(y_sm)
        y_k = np.array(y_k)
        if y_sm.shape != y_k.shape:
            print(f"  Shape mismatch: {y_sm.shape} vs {y_k.shape}", file=sys.stderr)
            all_ok = False
            continue
        max_abs = float(np.max(np.abs(y_sm - y_k)))
        ok = np.allclose(y_sm, y_k, atol=args.atol)
        print(f"  max|diff|={max_abs:.3e}  allclose(atol={args.atol})={ok}")
        if not ok:
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
