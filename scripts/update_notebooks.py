#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

REPLACEMENTS = [
    ("Tensorflow", "Keras (JAX backend)"),
    ("TensorFlow", "Keras (JAX backend)"),
    ("tensorflow", "keras"),
    (".h5", ".keras"),
    ("endswith('.h5')", "endswith('.keras')"),
    ('endswith(".h5")', 'endswith(".keras")'),
]

def apply_replacements_text(s: str) -> str:
    out = s
    for old, new in REPLACEMENTS:
        out = out.replace(old, new)
    return out

def process_notebook(path: Path) -> bool:
    data = json.loads(path.read_text())
    changed = False
    for cell in data.get("cells", []):
        if "source" in cell:
            src = cell["source"]
            if isinstance(src, list):
                joined = "".join(src)
                new_src = apply_replacements_text(joined)
                if new_src != joined:
                    cell["source"] = [new_src]
                    changed = True
            elif isinstance(src, str):
                new_src = apply_replacements_text(src)
                if new_src != src:
                    cell["source"] = new_src
                    changed = True
    if changed:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        Path(path).write_text(json.dumps(data, ensure_ascii=False))
    return changed

def main():
    ap = argparse.ArgumentParser(description="Update notebook references from TensorFlow/.h5 to Keras/.keras")
    ap.add_argument("paths", nargs="*", help="Notebook files or directories. Default: docs/getting_started")
    ap.add_argument("--recursive", "-r", action="store_true", help="Recurse into directories")
    args = ap.parse_args()

    targets = []
    if args.paths:
        for p in args.paths:
            path = Path(p)
            if path.is_file() and path.suffix == ".ipynb":
                targets.append(path)
            elif path.is_dir():
                it = path.rglob("*.ipynb") if args.recursive else path.glob("*.ipynb")
                targets.extend(list(it))
    else:
        base = Path("docs/getting_started")
        targets = list(base.glob("*.ipynb"))

    edited = 0
    for nb in targets:
        try:
            if process_notebook(nb):
                print(f"Updated: {nb}")
                edited += 1
        except Exception as e:
            print(f"Failed to update {nb}: {e}")

    print(f"Done. Updated {edited} notebook(s).")

if __name__ == "__main__":
    main()
