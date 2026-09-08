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

    For a field value :math:`T_m(x)` at spatial cell ``x``, the running state
    stores the sample count ``n``, mean ``mu`` and centered sum of squares
    ``M2 = sum_m (T_m-mu)^2``. Two batches ``A`` and ``B`` are merged with

    ``mu = mu_A + delta * n_B/(n_A+n_B)``

    and

    ``M2 = M2_A + M2_B + delta^2*n_A*n_B/(n_A+n_B)``,

    where ``delta = mu_B-mu_A``. This is algebraically equivalent to direct
    accumulation but avoids retaining all Monte Carlo fields.
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

        denominator = self.count - ddof
        if denominator <= 0:
            raise ValueError(
                "variance is undefined because sample_count <= ddof; "
                f"got sample_count={self.count}, ddof={ddof}. "
                "Use ddof=0 for a one-sample deterministic diagnostic."
            )

        assert self.m2 is not None
        assert self.minimum is not None
        assert self.maximum is not None
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
