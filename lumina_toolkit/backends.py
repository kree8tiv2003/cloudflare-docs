"""Model backends for features that need learned models.

Features call `require_backend(kind)`. Users register an implementation
(ONNX, PyTorch, a cloud API...) with `register_backend`. Nothing is bundled.
"""
from typing import Any, Callable, Dict

_BACKENDS: Dict[str, Callable[..., Any]] = {}


class BackendMissing(RuntimeError):
    pass


def register_backend(kind: str, fn: Callable[..., Any]) -> None:
    _BACKENDS[kind] = fn


def require_backend(kind: str) -> Callable[..., Any]:
    try:
        return _BACKENDS[kind]
    except KeyError:
        raise BackendMissing(
            f"No '{kind}' backend registered. Call lumina_toolkit.backends."
            f"register_backend('{kind}', fn) with a model implementation."
        ) from None
