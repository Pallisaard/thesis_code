from collections.abc import Callable
from typing import Any

from jax import tree_map


def tree_key_map(func: Callable[[str], Any], tree: Any) -> Any:
    if isinstance(tree, dict):
        return {tree_key_map(func, k): v for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(func(v) for v in tree)
    else:
        return func(tree)


tree_map = tree_map
