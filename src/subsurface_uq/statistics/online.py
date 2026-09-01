from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class FieldStatistics:
    count: int
    mean: Array
    variance: Array
    std: Array
    minimum: Array
    maximum: Array


class OnlineFieldStatistics:
    """Streaming per-cell statistics using a batch-merge form of Welford.

    Temperature fields are two-dimensional. ``update`` accepts either one
    ``[H,W]`` field or a batch ``[B,H,W]`` and stores only aggregate statistics.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean: Array | None = None
        self.m2: Array | None = None
        self.minimum: Array | None = None
        self.maximum: Array | None = None

    @property
    def field_shape(self) -> tuple[int, int] | None:
        return None if self.mean is None else tuple(int(v) for v in self.mean.shape)

    def update(self, values: Array) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim == 2:
            batch = batch[None, ...]
        if batch.ndim != 3:
            raise ValueError(f"expected [H,W] or [B,H,W], got {batch.shape}")
        if batch.shape[0] == 0:
            return
        if not np.all(np.isfinite(batch)):
            raise ValueError("statistics input contains non-finite values")

        shape = tuple(int(v) for v in batch.shape[1:])
        if self.mean is not None and self.mean.shape != shape:
            raise ValueError(
                f"field shape changed from {self.mean.shape} to {shape}"
            )

        batch_count = int(batch.shape[0])
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        batch_min = batch.min(axis=0)
        batch_max = batch.max(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.minimum = batch_min
            self.maximum = batch_max
            return

        assert self.mean is not None
        assert self.m2 is not None
        assert self.minimum is not None
        assert self.maximum is not None

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2
            + batch_m2
            + delta * delta * (self.count * batch_count / total)
        )
        self.minimum = np.minimum(self.minimum, batch_min)
        self.maximum = np.maximum(self.maximum, batch_max)
        self.count = total

    def finalize(self, *, ddof: int = 1) -> FieldStatistics:
        if self.count == 0 or self.mean is None:
            raise RuntimeError("no fields have been accumulated")
        if ddof < 0:
            raise ValueError("ddof must be non-negative")

        assert self.m2 is not None
        assert self.minimum is not None
        assert self.maximum is not None
        denominator = self.count - ddof
        if denominator <= 0:
            variance = np.zeros_like(self.mean)
        else:
            variance = self.m2 / denominator
        variance = np.maximum(variance, 0.0)
        return FieldStatistics(
            count=self.count,
            mean=self.mean.astype(np.float32),
            variance=variance.astype(np.float32),
            std=np.sqrt(variance).astype(np.float32),
            minimum=self.minimum.astype(np.float32),
            maximum=self.maximum.astype(np.float32),
        )
