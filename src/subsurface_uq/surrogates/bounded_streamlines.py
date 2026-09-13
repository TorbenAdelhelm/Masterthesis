from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

Array = np.ndarray


class StreamlineIntegrationError(RuntimeError):
    """Raised when a bounded streamline exceeds the configured RHS budget."""


@dataclass
class BoundedStreamlineFactory:
    """Drop-in replacement for release25 ``make_streamlines`` used by UQ runs.

    The implementation preserves release25's velocity interpolation, RK solver,
    integration horizon, output sampling, heat-pump offsets, and fading rule.
    The only numerical change is that integration terminates once a trajectory
    leaves the valid grid. A configurable RHS-evaluation watchdog prevents one
    pathological trajectory from appearing to hang indefinitely.
    """

    max_nfev: int = 100_000
    diagnostics: bool = False
    slow_streamline_seconds: float = 2.0
    call_count: int = 0

    def __post_init__(self) -> None:
        self.max_nfev = int(self.max_nfev)
        if self.max_nfev < 0:
            raise ValueError("max_nfev must be non-negative; use 0 to disable the watchdog")
        self.slow_streamline_seconds = float(self.slow_streamline_seconds)
        if self.slow_streamline_seconds < 0.0:
            raise ValueError("slow_streamline_seconds must be non-negative")

    @staticmethod
    def _pass_name(offset: float | int | None) -> str:
        if offset is None or float(offset) == 0.0:
            return "center"
        if float(offset) > 0.0:
            return f"offset+{float(offset):g}"
        return f"offset{float(offset):g}"

    def __call__(
        self,
        *,
        mat_ids: Array,
        vx: Array,
        vy: Array,
        dims: tuple[int, int],
        offset: float | int | None = None,
        randomK_data: bool = False,
        method: str = "RK45",
        **kwargs: Any,
    ) -> torch.Tensor:
        self.call_count += 1
        sample_index = (self.call_count - 1) // 3 + 1
        pass_name = self._pass_name(offset)
        return bounded_make_streamlines(
            mat_ids=mat_ids,
            vx=vx,
            vy=vy,
            dims=dims,
            offset=offset,
            randomK_data=randomK_data,
            method=method,
            max_nfev=self.max_nfev,
            diagnostics=self.diagnostics,
            slow_streamline_seconds=self.slow_streamline_seconds,
            sample_index=sample_index,
            pass_name=pass_name,
            **kwargs,
        )


def _velocity_rhs(
    x: Array,
    y: Array,
    vx: Array,
    vy: Array,
    *,
    random_k_data: bool,
):
    # Preserve release25's axis convention exactly.
    if random_k_data:
        fx = RegularGridInterpolator(
            (x, y), vx, bounds_error=False, fill_value=None, method="linear"
        )
        fy = RegularGridInterpolator(
            (x, y), vy, bounds_error=False, fill_value=None, method="linear"
        )
    else:
        fy = RegularGridInterpolator(
            (x, y), vx, bounds_error=False, fill_value=None, method="linear"
        )
        fx = RegularGridInterpolator(
            (x, y), vy, bounds_error=False, fill_value=None, method="linear"
        )

    def rhs(_t: float, state: Array) -> Array:
        return np.squeeze([fy(state), fx(state)])

    return rhs


def _boundary_events(x_max: float, y_max: float):
    def x_min(_t: float, state: Array) -> float:
        return float(state[0])

    def x_max_event(_t: float, state: Array) -> float:
        return float(x_max - state[0])

    def y_min(_t: float, state: Array) -> float:
        return float(state[1])

    def y_max_event(_t: float, state: Array) -> float:
        return float(y_max - state[1])

    events = (x_min, x_max_event, y_min, y_max_event)
    for event in events:
        event.terminal = True
        # All four functions decrease when crossing from inside to outside.
        event.direction = -1.0
    return events


def _bounded_calc_streamline(
    rhs,
    maxs_xy: tuple[float, float],
    start: Array,
    *,
    t_end: float = 27.5,
    t_steps: int = 10_000,
    method: str = "RK45",
    max_nfev: int = 100_000,
    diagnostic_prefix: str = "",
    **kwargs: Any,
) -> tuple[Array, Array, Array, int]:
    nfev = 0

    def limited_rhs(t: float, state: Array) -> Array:
        nonlocal nfev
        nfev += 1
        if max_nfev > 0 and nfev > max_nfev:
            raise StreamlineIntegrationError(
                f"{diagnostic_prefix} exceeded max_nfev={max_nfev}; "
                f"last_t={t:.6g}, last_state={np.asarray(state).tolist()}"
            )
        return rhs(t, state)

    sol = solve_ivp(
        limited_rhs,
        [0.0, float(t_end)],
        np.asarray(start, dtype=float),
        t_eval=np.linspace(0.0, float(t_end), int(t_steps)),
        method=method,
        events=_boundary_events(float(maxs_xy[0]), float(maxs_xy[1])),
        **kwargs,
    )
    if not sol.success:
        raise StreamlineIntegrationError(
            f"{diagnostic_prefix} solve_ivp failed after nfev={nfev}: {sol.message}"
        )

    sol_x = sol.y[0]
    sol_y = sol.y[1]

    # Keep release25's post-filter as a final numerical guard. In bounded mode
    # the terminal events should make these cuts a no-op except for roundoff.
    sol_x = sol_x[sol_x <= maxs_xy[0]]
    sol_y = sol_y[sol_y <= maxs_xy[1]]
    sol_x = sol_x[sol_x >= 0.0]
    sol_y = sol_y[sol_y >= 0.0]
    length = min(sol_x.shape[0], sol_y.shape[0])
    sol_x = sol_x[:length]
    sol_y = sol_y[:length]
    t = sol.t[:length]
    return sol_x, sol_y, t, nfev


def _draw_faded_streamlines(image_data: Array, streamlines: list[tuple[Array, Array, Array]]) -> Array:
    for sol_x, sol_y, t in streamlines:
        if len(t) > 0:
            val = t[::-1]
            maximum = float(np.max(val))
            val = val / maximum if maximum > 0.0 else np.ones_like(val)
        else:
            val = 1.0
        image_data[((sol_x + 0.5).astype(int), (sol_y + 0.5).astype(int))] = val
    return image_data


def bounded_make_streamlines(
    *,
    mat_ids: Array,
    vx: Array,
    vy: Array,
    dims: tuple[int, int],
    offset: float | int | None = None,
    randomK_data: bool = False,
    method: str = "RK45",
    max_nfev: int = 100_000,
    diagnostics: bool = False,
    slow_streamline_seconds: float = 2.0,
    sample_index: int | None = None,
    pass_name: str = "center",
    **kwargs: Any,
) -> torch.Tensor:
    """Release25-compatible streamline rasterization with domain termination."""

    mat_ids = np.asarray(mat_ids)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    dims = (int(dims[0]), int(dims[1]))
    if mat_ids.shape != dims or vx.shape != dims or vy.shape != dims:
        raise ValueError(
            "mat_ids, vx and vy must match dims; "
            f"got {mat_ids.shape}, {vx.shape}, {vy.shape}, dims={dims}"
        )
    if not (np.all(np.isfinite(vx)) and np.all(np.isfinite(vy))):
        raise ValueError("velocity fields must be finite")

    positions = np.array(np.where(mat_ids == 2)).T.astype(float)
    positions += np.array([0.5, 0.5])
    if offset is not None:
        positions += np.array([0.0, float(offset)])

    x = np.arange(0, dims[0])
    y = np.arange(0, dims[1])
    resolution = 5.0
    rhs = _velocity_rhs(
        x,
        y,
        vx / resolution,
        vy / resolution,
        random_k_data=randomK_data,
    )

    prefix = (
        f"sample={sample_index if sample_index is not None else '?'} "
        f"pass={pass_name}"
    )
    if diagnostics:
        print(
            f"[bounded-streamlines] {prefix} heat_pumps={positions.shape[0]} "
            f"method={method} max_nfev={max_nfev} "
            f"vx=[{float(np.min(vx)):.6g},{float(np.max(vx)):.6g}] "
            f"vy=[{float(np.min(vy)):.6g},{float(np.max(vy)):.6g}]"
        )

    started = perf_counter()
    streamlines: list[tuple[Array, Array, Array]] = []
    total_nfev = 0
    iterator = tqdm(
        enumerate(positions, start=1),
        total=positions.shape[0],
        desc=f"Calculating bounded streamlines ({pass_name})",
    )
    for hp_index, hp in iterator:
        hp_started = perf_counter()
        diagnostic_prefix = (
            f"{prefix} heat_pump={hp_index}/{positions.shape[0]} "
            f"start={np.asarray(hp).tolist()}"
        )
        sol_x, sol_y, t, nfev = _bounded_calc_streamline(
            rhs,
            (float(x.max()), float(y.max())),
            np.asarray(hp).T,
            t_end=27.5,
            t_steps=10_000,
            method=method,
            max_nfev=max_nfev,
            diagnostic_prefix=diagnostic_prefix,
            **kwargs,
        )
        total_nfev += nfev
        streamlines.append((sol_x, sol_y, t))
        elapsed = perf_counter() - hp_started
        if diagnostics and elapsed >= slow_streamline_seconds:
            print(
                f"[bounded-streamlines] slow {diagnostic_prefix} "
                f"elapsed={elapsed:.3f}s nfev={nfev} points={len(t)}"
            )

    raster = _draw_faded_streamlines(np.zeros(dims), streamlines)
    if diagnostics:
        print(
            f"[bounded-streamlines] finished {prefix} "
            f"elapsed={perf_counter() - started:.3f}s total_nfev={total_nfev}"
        )
    return torch.tensor(raster)


def configure_release25_streamlines(
    adapter: Any,
    *,
    mode: str,
    max_nfev: int = 100_000,
    diagnostics: bool = False,
    slow_streamline_seconds: float = 2.0,
) -> None:
    """Configure a loaded release25 adapter without modifying upstream code.

    ``release25`` leaves the imported upstream ``make_streamlines`` callable
    untouched. ``bounded`` swaps only that callable for the local compatible
    implementation, so ``Release25RuntimeAdapter._streamlines`` still performs
    the same center/+10/-10 sequence.
    """

    if mode not in {"release25", "bounded"}:
        raise ValueError("streamline mode must be 'release25' or 'bounded'")
    if mode == "release25":
        return
    adapter._make_streamlines = BoundedStreamlineFactory(
        max_nfev=max_nfev,
        diagnostics=diagnostics,
        slow_streamline_seconds=slow_streamline_seconds,
    )
