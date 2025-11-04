# Backends

Keras 3 supports multiple backends. stella works with:

- `jax` (default if installed)
- `torch`

To select a backend:

```bash
export KERAS_BACKEND=jax   # or torch
```

Inside Python:

```python
import keras
print(keras.backend.backend())
```
