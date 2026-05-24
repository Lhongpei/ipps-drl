"""Public inference API.

The MCTS submodule is imported lazily inside ``InferenceEngine`` because it
depends on the optional C++ env wrapper. Importing the package itself stays cheap
even if that extension is not built.
"""

from .engine import InferenceEngine
from .result import InferenceResult

__all__ = ["InferenceEngine", "InferenceResult"]
