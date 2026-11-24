from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Self


class PyTreeOptionsBase:
    """Shared PyTree serialization and immutable update methods for frozen option dataclasses."""

    _child_fields: ClassVar[tuple[str, ...]] = ()
    _aux_fields: ClassVar[tuple[str, ...]] = ()

    def replace(self, **changes: Any) -> Self:
        """Return a copy of the options with updated fields."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = tuple(getattr(self, f) for f in self._child_fields)
        aux_data = tuple(getattr(self, f) for f in self._aux_fields)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Self:
        kwargs = dict(zip(cls._child_fields, children, strict=True))
        kwargs.update(zip(cls._aux_fields, aux_data, strict=True))
        return cls(**kwargs)
