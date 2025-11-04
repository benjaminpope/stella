import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .version import __version__

# Keep package import lightweight: do not auto-import heavy submodules.
# Import submodules directly, e.g. `from stella.neural_network import ConvNN`.

# Convenience backend utilities exposed at top-level.
from .backends import check_backend, benchmark, swap_backend, require_backend  # noqa: F401
from . import models  # noqa: F401

__all__ = [
    "__version__",
    "check_backend",
    "benchmark",
    "swap_backend",
    "require_backend",
    "models",
]

try:
	from .backends import check_backend, benchmark
except Exception:
	pass
