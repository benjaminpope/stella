# Installation

Choose a backend and install dependencies:

```bash
# JAX (CPU)
export KERAS_BACKEND=jax
pip install -r requirements-jax.txt

# or PyTorch
export KERAS_BACKEND=torch
pip install -r requirements-torch.txt

# Install stella (editable recommended for development)
pip install -e .[dev]
```
