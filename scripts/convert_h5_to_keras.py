#!/usr/bin/env python3
import argparse
import sys
import os
import shutil
from pathlib import Path

def main():
    # Force TensorFlow backend to avoid JAX conflicts during conversion
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    p = argparse.ArgumentParser(description="Convert TF-Keras .h5 models or SavedModel directories to Keras (.keras) format using TF backend.")
    p.add_argument("paths", nargs="+", help="Files or directories to convert.")
    p.add_argument("--recursive", "-r", action="store_true", help="Recurse into directories.")
    p.add_argument("--out", "-o", default=None, help="Output directory (mirrors structure). Default: alongside source.")
    p.add_argument("--keep-tmp", action="store_true", help="Keep temporary SavedModel directories.")
    p.add_argument("--cadences", type=int, default=200, help="Assumed input length for canonical rebuild fallback (default: 200)")
    args = p.parse_args()

    try:
        import tensorflow as tf  # only needed to load legacy .h5
    except Exception:
        print("TensorFlow is required only for conversion. Install temporarily:", file=sys.stderr)
        print("  pip install 'tensorflow>=2.15'  # or tensorflow-macos/tensorflow-metal on Apple Silicon", file=sys.stderr)
        return 2

    try:
        import keras  # Keras 3 to write .keras format with TF backend
    except Exception:
        print("Keras (v3) is required to save the .keras format. Install:", file=sys.stderr)
        print("  pip install 'keras>=3.0.0'", file=sys.stderr)
        return 2

    def is_savedmodel_dir(d: Path) -> bool:
        return d.is_dir() and (d / "saved_model.pb").exists()

    # Collect targets: .h5 files and SavedModel directories
    targets = []
    for pth in args.paths:
        p = Path(pth)
        if p.is_file() and p.suffix.lower() == ".h5":
            targets.append(p)
        elif p.is_dir():
            # If the provided directory itself is a SavedModel, include it
            if is_savedmodel_dir(p):
                targets.append(p)
            if not args.recursive:
                for child in p.iterdir():
                    if child.is_file() and child.suffix.lower() == ".h5":
                        targets.append(child)
                    elif is_savedmodel_dir(child):
                        targets.append(child)
            else:
                targets.extend(list(p.rglob("*.h5")))
                for d in p.rglob("*/"):
                    if is_savedmodel_dir(d):
                        targets.append(d)
        else:
            print(f"Skipping non-existent path: {p}", file=sys.stderr)

    if not targets:
        print("No .h5 files or SavedModel directories found.", file=sys.stderr)
        return 0

    def build_canonical_tf_model(cadences: int):
        # Rebuild the default architecture used by stella for weight-only load
        from tensorflow import keras as tfk
        model = tfk.Sequential(name="stella_cnn")
        model.add(tfk.layers.Input(shape=(cadences, 1), name="input"))
        model.add(tfk.layers.Conv1D(16, 7, activation='relu', padding='same', name='conv1d'))
        model.add(tfk.layers.MaxPooling1D(2, name='max_pooling1d'))
        model.add(tfk.layers.Dropout(0.1, name='dropout'))
        model.add(tfk.layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d_1'))
        model.add(tfk.layers.MaxPooling1D(2, name='max_pooling1d_1'))
        model.add(tfk.layers.Dropout(0.1, name='dropout_1'))
        model.add(tfk.layers.Flatten(name='flatten'))
        model.add(tfk.layers.Dense(32, activation='relu', name='dense'))
        model.add(tfk.layers.Dropout(0.1, name='dropout_2'))
        model.add(tfk.layers.Dense(1, activation='sigmoid', name='dense_1'))
        # Build the model to ensure weights shapes are set
        model.build(input_shape=(None, cadences, 1))
        return model

    def build_canonical_keras_model(cadences: int):
        # Same architecture but using Keras 3 API (backend-agnostic)
        import keras as k
        model = k.Sequential(name="stella_cnn")
        model.add(k.layers.Input(shape=(cadences, 1), name="input"))
        model.add(k.layers.Conv1D(16, 7, activation='relu', padding='same', name='conv1d'))
        model.add(k.layers.MaxPooling1D(2, name='max_pooling1d'))
        model.add(k.layers.Dropout(0.1, name='dropout'))
        model.add(k.layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d_1'))
        model.add(k.layers.MaxPooling1D(2, name='max_pooling1d_1'))
        model.add(k.layers.Dropout(0.1, name='dropout_1'))
        model.add(k.layers.Flatten(name='flatten'))
        model.add(k.layers.Dense(32, activation='relu', name='dense'))
        model.add(k.layers.Dropout(0.1, name='dropout_2'))
        model.add(k.layers.Dense(1, activation='sigmoid', name='dense_1'))
        model.build(input_shape=(None, cadences, 1))
        return model

    def transfer_weights(tf_model, k_model):
        # Map weights by layer name where shapes match
        tf_layers = {layer.name: layer for layer in tf_model.layers}
        k_layers = {layer.name: layer for layer in k_model.layers}
        assigned, skipped = 0, 0
        for name, tf_layer in tf_layers.items():
            if name not in k_layers:
                skipped += 1
                continue
            k_layer = k_layers[name]
            try:
                tf_w = tf_layer.get_weights()
                k_w = k_layer.get_weights()
                if len(tf_w) != len(k_w):
                    skipped += 1
                    continue
                # Check shapes
                if any(tw.shape != kw.shape for tw, kw in zip(tf_w, k_w)):
                    skipped += 1
                    continue
                k_layer.set_weights(tf_w)
                assigned += 1
            except Exception:
                skipped += 1
                continue
        print(f"Transferred weights: assigned {assigned} layers, skipped {skipped}.")

    def extract_savedmodel_weights(sm_path: Path):
        # Load TensorFlow SavedModel and extract layer weights by name
        import re
        var_map = {}
        try:
            obj = tf.saved_model.load(str(sm_path))
        except Exception as e:
            print(f"tf.saved_model.load failed for {sm_path}: {e}", file=sys.stderr)
            return var_map

        # Variables may be under .variables or .trainable_variables
        variables = []
        if hasattr(obj, 'variables'):
            variables.extend(list(obj.variables))
        if hasattr(obj, 'trainable_variables'):
            variables.extend(list(obj.trainable_variables))
        # Deduplicate
        seen = set()
        uniq_vars = []
        for v in variables:
            if v.ref() in seen:
                continue
            seen.add(v.ref())
            uniq_vars.append(v)

        pattern = re.compile(r"(?:.*/)?(?P<layer>[^/]+)/(?P<weight>kernel|bias)(?::\d+)?$")
        for v in uniq_vars:
            name = getattr(v, 'name', '')
            m = pattern.match(name)
            if not m:
                continue
            layer = m.group('layer')
            weight = m.group('weight')
            try:
                val = v.numpy()
            except Exception:
                continue
            var_map.setdefault(layer, {})[weight] = val
        return var_map

    for src in targets:
        if is_savedmodel_dir(src):
            # Extract weights via tf.saved_model and assign to Keras 3 canonical model
            var_map = extract_savedmodel_weights(src)
            if not var_map:
                print(f"No usable weights found in {src}", file=sys.stderr)
                continue

            import numpy as np
            k_model = build_canonical_keras_model(args.cadences)
            assigned, skipped = 0, 0
            for lname in ["conv1d", "conv1d_1", "dense", "dense_1"]:
                layer = None
                for l in k_model.layers:
                    if l.name == lname:
                        layer = l
                        break
                if layer is None:
                    skipped += 1
                    continue
                want = []
                if hasattr(layer, 'kernel') or any(w.name.endswith('kernel:0') for w in layer.weights):
                    if var_map.get(lname, {}).get('kernel') is not None:
                        want.append(var_map[lname]['kernel'])
                    else:
                        skipped += 1
                        continue
                if hasattr(layer, 'bias') or any(w.name.endswith('bias:0') for w in layer.weights):
                    b = var_map.get(lname, {}).get('bias')
                    if b is not None:
                        want.append(b)
                # Validate shapes
                cur_shapes = [w.shape for w in layer.get_weights()]
                want_shapes = [np.array(w).shape for w in want]
                if cur_shapes and cur_shapes == want_shapes:
                    try:
                        layer.set_weights(want)
                        assigned += 1
                    except Exception:
                        skipped += 1
                else:
                    skipped += 1
            print(f"Transferred from SavedModel: assigned {assigned} layers, skipped {skipped}.")

            if args.out:
                dst_dir = Path(args.out)
                dst = dst_dir / (src.name + ".keras")
            else:
                dst_dir = src.parent
                dst = dst_dir / (src.name + ".keras")

            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                k_model.save(str(dst))
                print(f"Converted SavedModel -> {dst}")
            except Exception as e:
                print(f"Failed to save {src} -> {dst}: {e}", file=sys.stderr)
            continue

        # Otherwise treat as legacy .h5
        # Try direct Keras load first (may handle legacy H5 more flexibly)
        k_model = None
        try:
            k_model = keras.models.load_model(str(src), compile=False, safe_mode=False)
            print(f"Loaded via Keras: {src}")
        except Exception as e1:
            print(f"Keras direct load failed for {src}: {e1}")
            # Fallback: load with tf.keras
            try:
                tf_model = tf.keras.models.load_model(str(src), compile=False)
            except Exception as e2:
                print(f"Failed to load with tf.keras {src}: {e2}", file=sys.stderr)
                # Final fallback: rebuild canonical TF model and load weights only
                try:
                    tf_model = build_canonical_tf_model(args.cadences)
                    # Try loading by order; Keras 3 API uses skip_mismatch to ignore optimizer/mismatch
                    tf_model.load_weights(str(src), skip_mismatch=True)
                    print(f"Loaded weights only into canonical model for {src}")
                except Exception as e3:
                    print(f"Failed weight-only load for {src}: {e3}", file=sys.stderr)
                    continue

            # Transfer weights from tf.keras model to Keras 3 canonical model
            k_model = build_canonical_keras_model(args.cadences)
            transfer_weights(tf_model, k_model)

        if args.out:
            dst_dir = Path(args.out)
            dst = dst_dir / src.with_suffix(".keras").name
        else:
            dst_dir = src.parent
            dst = src.with_suffix(".keras")

        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            k_model.save(str(dst))
            print(f"Converted: {src} -> {dst}")
        except Exception as e:
            print(f"Failed to save {src} -> {dst}: {e}", file=sys.stderr)
        finally:
            pass

    return 0

if __name__ == "__main__":
    sys.exit(main())
