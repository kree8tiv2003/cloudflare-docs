from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class Feature:
    key: str
    title: str
    description: str
    run: Callable
    needs_backend: str = ""  # empty = works out of the box


FEATURES: Dict[str, Feature] = {}


def feature(key, title, description, needs_backend=""):
    def deco(fn):
        FEATURES[key] = Feature(key, title, description, fn, needs_backend)
        return fn
    return deco


def get_feature(key: str) -> Feature:
    _load()
    return FEATURES[key]


def list_features() -> List[Feature]:
    _load()
    return list(FEATURES.values())


def _load():
    from . import features  # noqa: F401  (import registers features)
