from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np

Array = np.ndarray


@runtime_checkable
class PermeabilitySampler(Protocol):
    """Source of permeability batches shaped ``[B, H, W]``."""

    def __iter__(self) -> Iterator[Array]: ...

    def __len__(self) -> int: ...
