from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass(frozen=True)
class RQ1Config:
    experiment_id: str
    output_root: Path
    release25_repo: Path
    cnn1_dir: Path
    cnn2_dir: Path
    prepared_pki_dir: Path
    fixed_run_id: str
    device: str
    cell_size_m: float
    background_temperature: float
    budgets: tuple[int, ...]
    comparison_budgets: tuple[int, ...]
    repetitions: int
    main_seed: int
    repetition_seeds: tuple[int, ...]
    batch_size: int
    quantiles: tuple[float, float, float]
    receptors: tuple[tuple[int, int], ...]
    mean_anomaly_roi: tuple[int, int, int, int] | None
    quantile_chunk_rows: int
    input_preview_count: int
    mean_log10_k: float
    std_log10_k: float
    length_scale_y_m: float
    length_scale_x_m: float
    n_modes: int | None
    energy_threshold: float | None
    observations: tuple[tuple[int, int, float], ...]
    observation_std_log10_k: float
    conditioning_tolerance_log10: float | None
    streamline_mode: str
    streamline_method: str
    streamline_max_nfev: int
    streamline_diagnostics: bool
    streamline_slow_seconds: float
    random_k: bool
    retain_scratch: bool

    @property
    def max_budget(self) -> int:
        return self.budgets[-1]

    @property
    def max_comparison_budget(self) -> int:
        return self.comparison_budgets[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment": {
                "id": self.experiment_id,
                "output_root": str(self.output_root),
            },
            "release25": {
                "repo": str(self.release25_repo),
                "cnn1_dir": str(self.cnn1_dir),
                "cnn2_dir": str(self.cnn2_dir),
                "prepared_pki_dir": str(self.prepared_pki_dir),
                "fixed_run_id": self.fixed_run_id,
                "device": self.device,
                "cell_size_m": self.cell_size_m,
                "background_temperature": self.background_temperature,
                "random_k": self.random_k,
            },
            "sampling": {
                "budgets": list(self.budgets),
                "comparison_budgets": list(self.comparison_budgets),
                "repetitions": self.repetitions,
                "main_seed": self.main_seed,
                "repetition_seeds": list(self.repetition_seeds),
                "batch_size": self.batch_size,
                "quantiles": list(self.quantiles),
                "quantile_chunk_rows": self.quantile_chunk_rows,
                "input_preview_count": self.input_preview_count,
            },
            "qoi": {
                "receptors": [list(value) for value in self.receptors],
                "mean_anomaly_roi": (
                    None if self.mean_anomaly_roi is None else list(self.mean_anomaly_roi)
                ),
            },
            "grf": {
                "mean_log10_k": self.mean_log10_k,
                "std_log10_k": self.std_log10_k,
                "length_scale_y_m": self.length_scale_y_m,
                "length_scale_x_m": self.length_scale_x_m,
                "n_modes": self.n_modes,
                "energy_threshold": self.energy_threshold,
                "observations": [list(value) for value in self.observations],
                "observation_std_log10_k": self.observation_std_log10_k,
                "conditioning_tolerance_log10": self.conditioning_tolerance_log10,
            },
            "streamlines": {
                "mode": self.streamline_mode,
                "method": self.streamline_method,
                "max_nfev": self.streamline_max_nfev,
                "diagnostics": self.streamline_diagnostics,
                "slow_seconds": self.streamline_slow_seconds,
            },
            "storage": {
                "retain_scratch": self.retain_scratch,
            },
        }


def load_rq1_config(path: str | Path) -> RQ1Config:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "config")

    experiment = _mapping(root.get("experiment"), "experiment")
    release25 = _mapping(root.get("release25"), "release25")
    sampling = _mapping(root.get("sampling"), "sampling")
    qoi = _mapping(root.get("qoi"), "qoi")
    grf = _mapping(root.get("grf"), "grf")
    streamlines = _mapping(root.get("streamlines", {}), "streamlines")
    storage = _mapping(root.get("storage", {}), "storage")

    experiment_id = str(experiment.get("id", "")).strip()
    if not experiment_id:
        raise ValueError("experiment.id must be non-empty")

    budgets = tuple(int(v) for v in sampling.get("budgets", (32, 64, 128, 256, 512, 1024)))
    comparison = tuple(
        int(v) for v in sampling.get("comparison_budgets", (32, 64, 128, 256, 512))
    )
    if not budgets or sorted(set(budgets)) != list(budgets):
        raise ValueError("sampling.budgets must be unique and strictly increasing")
    if not comparison or sorted(set(comparison)) != list(comparison):
        raise ValueError("sampling.comparison_budgets must be unique and strictly increasing")
    if any(not _power_of_two(v) for v in budgets + comparison):
        raise ValueError("RQ1 sampling budgets must be powers of two")
    if not set(comparison).issubset(set(budgets)):
        raise ValueError("comparison_budgets must be a subset of budgets")

    repetitions = int(sampling.get("repetitions", 4))
    if repetitions <= 0:
        raise ValueError("sampling.repetitions must be positive")
    main_seed = int(sampling.get("main_seed", 2907))
    seeds_raw = sampling.get("repetition_seeds")
    if seeds_raw is None:
        raise ValueError("sampling.repetition_seeds must be provided explicitly")
    repetition_seeds = tuple(int(v) for v in seeds_raw)
    if len(repetition_seeds) != repetitions or len(set(repetition_seeds)) != repetitions:
        raise ValueError(
            "sampling.repetition_seeds must contain one unique seed per repetition"
        )
    if main_seed in repetition_seeds:
        raise ValueError(
            "sampling.main_seed must differ from repetition seeds so the empirical "
            "reference is independent of repeated MC/RQMC comparisons"
        )

    quantiles = tuple(float(v) for v in sampling.get("quantiles", (0.05, 0.5, 0.95)))
    if len(quantiles) != 3 or not np.allclose(quantiles, (0.05, 0.5, 0.95)):
        raise ValueError("RQ1 quantiles are fixed to [0.05, 0.5, 0.95]")

    receptors_raw = qoi.get("receptors", [])
    receptors: list[tuple[int, int]] = []
    for item in receptors_raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each qoi.receptors entry must be [row, col]")
        row, col = int(item[0]), int(item[1])
        if row < 0 or col < 0:
            raise ValueError("receptor indices must be non-negative")
        receptors.append((row, col))

    roi_raw = qoi.get("mean_anomaly_roi")
    roi: tuple[int, int, int, int] | None
    if roi_raw is None:
        roi = None
    else:
        if not isinstance(roi_raw, (list, tuple)) or len(roi_raw) != 4:
            raise ValueError(
                "qoi.mean_anomaly_roi must be null or "
                "[row_start,row_stop,col_start,col_stop]"
            )
        roi_values = tuple(int(v) for v in roi_raw)
        if (
            roi_values[0] < 0
            or roi_values[2] < 0
            or roi_values[1] <= roi_values[0]
            or roi_values[3] <= roi_values[2]
        ):
            raise ValueError(
                "qoi.mean_anomaly_roi must define a non-empty half-open window"
            )
        roi = (
            roi_values[0],
            roi_values[1],
            roi_values[2],
            roi_values[3],
        )

    n_modes_raw = grf.get("n_modes")
    energy_raw = grf.get("energy_threshold")
    if (n_modes_raw is None) == (energy_raw is None):
        raise ValueError("grf must define exactly one of n_modes or energy_threshold")
    n_modes = None if n_modes_raw is None else int(n_modes_raw)
    energy = None if energy_raw is None else float(energy_raw)

    observations: list[tuple[int, int, float]] = []
    for item in grf.get("observations", []):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("each grf.observations entry must be [row, col, K_m2]")
        row, col, permeability = int(item[0]), int(item[1]), float(item[2])
        if row < 0 or col < 0 or not np.isfinite(permeability) or permeability <= 0.0:
            raise ValueError("GRF observations require non-negative cells and positive finite K")
        observations.append((row, col, permeability))
    if not observations:
        raise ValueError(
            "RQ1 requires at least one GRF observation because conditional GRF/KL MC "
            "is the primary experiment variant"
        )

    tolerance_raw = grf.get("conditioning_tolerance_log10")
    tolerance = None if tolerance_raw is None else float(tolerance_raw)
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0.0):
        raise ValueError("conditioning_tolerance_log10 must be positive when supplied")

    cell_size_m = float(release25.get("cell_size_m", 5.0))
    background = float(release25.get("background_temperature", 10.6))
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("release25.cell_size_m must be finite and positive")
    if not np.isfinite(background):
        raise ValueError("release25.background_temperature must be finite")

    std_log10_k = float(grf["std_log10_k"])
    ly = float(grf["length_scale_y_m"])
    lx = float(grf["length_scale_x_m"])
    if std_log10_k <= 0.0 or ly <= 0.0 or lx <= 0.0:
        raise ValueError("GRF standard deviation and length scales must be positive")

    mode = str(streamlines.get("mode", "bounded"))
    method = str(streamlines.get("method", "RK45"))
    if mode not in {"bounded", "release25"}:
        raise ValueError("streamlines.mode must be bounded or release25")
    if method not in {"RK45", "RK23", "Radau"}:
        raise ValueError("streamlines.method must be RK45, RK23 or Radau")

    config = RQ1Config(
        experiment_id=experiment_id,
        output_root=Path(str(experiment["output_root"])).expanduser().resolve(),
        release25_repo=Path(str(release25["repo"])).expanduser().resolve(),
        cnn1_dir=Path(str(release25["cnn1_dir"])).expanduser().resolve(),
        cnn2_dir=Path(str(release25["cnn2_dir"])).expanduser().resolve(),
        prepared_pki_dir=Path(str(release25["prepared_pki_dir"])).expanduser().resolve(),
        fixed_run_id=str(release25["fixed_run_id"]),
        device=str(release25.get("device", "cpu")),
        cell_size_m=cell_size_m,
        background_temperature=background,
        budgets=budgets,
        comparison_budgets=comparison,
        repetitions=repetitions,
        main_seed=main_seed,
        repetition_seeds=repetition_seeds,
        batch_size=int(sampling.get("batch_size", 1)),
        quantiles=(quantiles[0], quantiles[1], quantiles[2]),
        receptors=tuple(receptors),
        mean_anomaly_roi=roi,
        quantile_chunk_rows=int(sampling.get("quantile_chunk_rows", 16)),
        input_preview_count=int(sampling.get("input_preview_count", 3)),
        mean_log10_k=float(grf["mean_log10_k"]),
        std_log10_k=std_log10_k,
        length_scale_y_m=ly,
        length_scale_x_m=lx,
        n_modes=n_modes,
        energy_threshold=energy,
        observations=tuple(observations),
        observation_std_log10_k=float(grf.get("observation_std_log10_k", 0.0)),
        conditioning_tolerance_log10=tolerance,
        streamline_mode=mode,
        streamline_method=method,
        streamline_max_nfev=int(streamlines.get("max_nfev", 100000)),
        streamline_diagnostics=bool(streamlines.get("diagnostics", False)),
        streamline_slow_seconds=float(streamlines.get("slow_seconds", 2.0)),
        random_k=bool(release25.get("random_k", True)),
        retain_scratch=bool(storage.get("retain_scratch", False)),
    )
    if config.batch_size <= 0 or config.quantile_chunk_rows <= 0:
        raise ValueError("batch_size and quantile_chunk_rows must be positive")
    if config.input_preview_count < 0:
        raise ValueError("input_preview_count must be non-negative")
    if config.streamline_max_nfev < 0:
        raise ValueError("streamline max_nfev must be non-negative")
    if config.observation_std_log10_k < 0.0:
        raise ValueError("observation_std_log10_k must be non-negative")
    return config
