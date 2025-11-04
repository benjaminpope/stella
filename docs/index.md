# stella

Find stellar flares in TESS 2-min data with a CNN.

- Backends: set `KERAS_BACKEND` to `jax` (default) or `torch`.
- Packaged models: discover with `from stella import models as sm; sm.models`.
- Quickstart notebooks are under Getting Started.

```bash
export KERAS_BACKEND=jax  # or torch
pip install -e .[dev]
```
