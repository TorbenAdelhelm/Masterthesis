from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import shutil
from typing import Callable, Mapping

import numpy as np

from ..io import git_head, runtime_versions, sha256_file
from ..propagation import MonteCarloRunner
from ..sampling import (
    RELEASE25_PERLIN_FREQUENCY,
    RELEASE25_PERLIN_K_MAX,
    RELEASE25_PERLIN_K_MIN,
    ConditionalKLLogGaussianPermeabilityMap,
    DiagnosticPermeabilitySampler,
    GaussianCoordinatePermeabilitySampler,
    KLLogGaussianPermeabilityMap,
    PerlinCoordinatePermeabilityMap,
    PermeabilityDiagnostics,
    ScrambledSobolGaussianPermeabilitySampler,
    UniformCoordinatePermeabilitySampler,
)
from ..surrogates import Release25Surrogate
from ..surrogates.bounded_streamlines import configure_release25_streamlines
from ..surrogates.release25_runtime import Release25Runtime
from ..visualization import plot_permeability_diagnostics
from ..visualization.rq1 import (
    plot_mc_rqmc_comparison,
    plot_rq1_convergence,
    plot_rq1_qoi_distributions,
    plot_rq1_temperature_fields,
)
from .accumulators import (
    DiskTemperatureStoreAccumulator,
    RQ1QoIAccumulator,
    RQ1QoISamples,
    ReferenceConvergenceFieldStatisticsAccumulator,
)
from .analysis import (
    global_uncertainty_metrics,
    prefix_field_convergence,
    scalar_prefix_convergence,
    scalar_summary,
    write_temperature_field_products,
)
from .config import RQ1Config
from .io import write_csv, write_json, write_yaml

Array = np.ndarray

RQ1_WORDING = "How much permeability input uncertainty reaches the predicted temperature field?"


@dataclass
class _MainVariant:
    key: str
    method: str
    sampler: object
    diagnostics: PermeabilityDiagnostics


def _build_grf_maps(
    config: RQ1Config,
    shape: tuple[int, int],
) -> tuple[KLLogGaussianPermeabilityMap, ConditionalKLLogGaussianPermeabilityMap]:
    domain_size_m = (
        float(shape[0]) * config.cell_size_m,
        float(shape[1]) * config.cell_size_m,
    )
    prior = KLLogGaussianPermeabilityMap(
        shape=shape,
        domain_size_m=domain_size_m,
        mean_log10_k=config.mean_log10_k,
        std_log10_k=config.std_log10_k,
        length_scale_m=(config.length_scale_y_m, config.length_scale_x_m),
        n_modes=config.n_modes,
        energy_threshold=(
            config.energy_threshold if config.energy_threshold is not None else 0.95
        ),
    )
    indices = np.asarray([[row, col] for row, col, _ in config.observations], dtype=int)
    values = np.asarray([value for _, _, value in config.observations], dtype=np.float64)
    conditional = ConditionalKLLogGaussianPermeabilityMap.from_permeability_observations(
        prior=prior,
        observation_indices=indices,
        observation_k=values,
        observation_std_log10_k=config.observation_std_log10_k,
    )
    return prior, conditional


def _diagnostics(
    config: RQ1Config,
    *,
    conditional: bool,
) -> PermeabilityDiagnostics:
    indices = None
    log_values = None
    if conditional:
        indices = np.asarray(
            [[row, col] for row, col, _ in config.observations],
            dtype=np.int64,
        )
        log_values = np.log10(
            np.asarray([value for _, _, value in config.observations], dtype=np.float64)
        )
    return PermeabilityDiagnostics(
        preview_count=config.input_preview_count,
        ddof=1,
        training_k_range=(RELEASE25_PERLIN_K_MIN, RELEASE25_PERLIN_K_MAX),
        observation_indices=indices,
        observation_log10_k=log_values,
    )


def _diagnostics_payload(result, config: RQ1Config) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample_count": int(result.count),
        "space": "log10(K)",
        "spatial_mean_log10_k": float(np.mean(result.mean_log10_k)),
        "spatial_mean_std_log10_k": float(np.mean(result.std_log10_k)),
        "spatial_max_std_log10_k": float(np.max(result.std_log10_k)),
        "global_k_min_m2": result.minimum_k,
        "global_k_max_m2": result.maximum_k,
        "release25_training_range_m2": [
            RELEASE25_PERLIN_K_MIN,
            RELEASE25_PERLIN_K_MAX,
        ],
        "outside_training_fraction": result.outside_training_fraction,
        "outside_training_fraction_by_sample": list(
            result.outside_training_fraction_by_sample
        ),
        "conditioning_max_abs_log10_residual": (
            result.conditioning_max_abs_log10_residual
        ),
        "conditioning_rms_log10_residual": result.conditioning_rms_log10_residual,
        "conditioning_max_abs_log10_residual_by_sample": list(
            result.conditioning_max_abs_log10_residual_by_sample
        ),
        "conditioning_tolerance_log10": config.conditioning_tolerance_log10,
    }
    if (
        config.conditioning_tolerance_log10 is not None
        and result.conditioning_max_abs_log10_residual is not None
    ):
        payload["conditioning_within_tolerance"] = bool(
            result.conditioning_max_abs_log10_residual
            < config.conditioning_tolerance_log10
        )
    else:
        payload["conditioning_within_tolerance"] = None
    return payload


def _qoi_reference(samples: RQ1QoISamples) -> dict[str, dict[str, float | int]]:
    reference: dict[str, dict[str, float | int]] = {}
    if samples.mean_anomaly is not None:
        reference["mean_anomaly"] = scalar_summary(samples.mean_anomaly)
    for index, location in enumerate(samples.receptor_indices):
        reference[f"receptor_{index + 1:03d}"] = {
            **scalar_summary(samples.receptor_values[:, index]),
            "row": int(location[0]),
            "col": int(location[1]),
        }
    return reference


def _qoi_rows(
    *,
    experiment_id: str,
    variant: str,
    method: str,
    repetition: str | int,
    samples: RQ1QoISamples,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample_count = int(samples.receptor_values.shape[0])
    for index in range(sample_count):
        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "variant": variant,
            "method": method,
            "repetition": repetition,
            "sample_index": index + 1,
        }
        if samples.mean_anomaly is not None:
            row["mean_anomaly"] = float(samples.mean_anomaly[index])
        for receptor_index in range(samples.receptor_values.shape[1]):
            row[f"receptor_{receptor_index + 1:03d}"] = float(
                samples.receptor_values[index, receptor_index]
            )
        rows.append(row)
    return rows


def _receptor_metric_rows(
    *,
    variant: str,
    method: str,
    repetition: str | int,
    budget: int,
    samples: RQ1QoISamples,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = int(budget)
    for index, location in enumerate(samples.receptor_indices):
        summary = scalar_summary(samples.receptor_values[:n, index])
        rows.append(
            {
                "variant": variant,
                "method": method,
                "repetition": repetition,
                "budget": n,
                "receptor": f"receptor_{index + 1:03d}",
                "row": int(location[0]),
                "col": int(location[1]),
                **summary,
            }
        )
    return rows


def _main_convergence_rows(
    *,
    variant: str,
    method: str,
    field_points: list[dict[str, float | int]],
    global_reference: Mapping[str, float],
    qoi_samples: RQ1QoISamples,
    qoi_reference: Mapping[str, Mapping[str, float | int]],
    checkpoints: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in field_points:
        n = int(point["budget"])
        for statistic in ("e_mu", "e_sigma"):
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "repetition": "main",
                    "budget": n,
                    "target": "temperature_field",
                    "statistic": statistic,
                    "estimate": "",
                    "reference": "",
                    "error": float(point[statistic]),
                    "abs_error": float(point[statistic]),
                }
            )
        for statistic in ("U_RMS", "U95", "mean_std", "max_std"):
            estimate = float(point[statistic])
            reference = float(global_reference[statistic])
            error = estimate - reference
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "repetition": "main",
                    "budget": n,
                    "target": "global_uncertainty",
                    "statistic": statistic,
                    "estimate": estimate,
                    "reference": reference,
                    "error": error,
                    "abs_error": abs(error),
                }
            )

    scalar_targets: list[tuple[str, Array]] = []
    if qoi_samples.mean_anomaly is not None:
        scalar_targets.append(("mean_anomaly", qoi_samples.mean_anomaly))
    for index in range(qoi_samples.receptor_values.shape[1]):
        scalar_targets.append(
            (
                f"receptor_{index + 1:03d}",
                qoi_samples.receptor_values[:, index],
            )
        )
    for target, values in scalar_targets:
        for row in scalar_prefix_convergence(
            values,
            checkpoints=checkpoints,
            reference=qoi_reference[target],
            target=target,
        ):
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "repetition": "main",
                    **row,
                }
            )
    return rows


def _repeated_convergence_rows(
    *,
    variant: str,
    method: str,
    repetition: int,
    field_accumulator: ReferenceConvergenceFieldStatisticsAccumulator,
    global_reference: Mapping[str, float],
    qoi_samples: RQ1QoISamples,
    qoi_reference: Mapping[str, Mapping[str, float | int]],
    checkpoints: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in field_accumulator.points:
        for statistic, value in (("e_mu", point.e_mu), ("e_sigma", point.e_sigma)):
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "repetition": repetition,
                    "budget": point.budget,
                    "target": "temperature_field",
                    "statistic": statistic,
                    "estimate": "",
                    "reference": "",
                    "error": value,
                    "abs_error": value,
                }
            )
        for statistic in ("U_RMS", "U95", "mean_std", "max_std"):
            estimate = float(getattr(point, statistic.lower()) if statistic != "U_RMS" else point.u_rms)
            if statistic == "U95":
                estimate = point.u95
            elif statistic == "mean_std":
                estimate = point.mean_std
            elif statistic == "max_std":
                estimate = point.max_std
            reference = float(global_reference[statistic])
            error = estimate - reference
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "repetition": repetition,
                    "budget": point.budget,
                    "target": "global_uncertainty",
                    "statistic": statistic,
                    "estimate": estimate,
                    "reference": reference,
                    "error": error,
                    "abs_error": abs(error),
                }
            )

    scalar_targets: list[tuple[str, Array]] = []
    if qoi_samples.mean_anomaly is not None:
        scalar_targets.append(("mean_anomaly", qoi_samples.mean_anomaly))
    for index in range(qoi_samples.receptor_values.shape[1]):
        scalar_targets.append(
            (
                f"receptor_{index + 1:03d}",
                qoi_samples.receptor_values[:, index],
            )
        )
    for target, values in scalar_targets:
        for row in scalar_prefix_convergence(
            values,
            checkpoints=checkpoints,
            reference=qoi_reference[target],
            target=target,
        ):
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    "repetition": repetition,
                    **row,
                }
            )
    return rows


def _aggregate_mc_rqmc(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str, str, str], list[float]] = {}
    for row in rows:
        if row["repetition"] == "main":
            continue
        method = str(row["method"])
        if method not in {"MC", "RQMC"}:
            continue
        key = (
            int(row["budget"]),
            str(row["target"]),
            str(row["statistic"]),
            method,
        )
        grouped.setdefault(key, []).append(float(row["error"]))

    pairs = sorted(
        {
            (budget, target, statistic)
            for budget, target, statistic, _ in grouped
        }
    )
    result: list[dict[str, object]] = []
    for budget, target, statistic in pairs:
        mc = grouped.get((budget, target, statistic, "MC"), [])
        rqmc = grouped.get((budget, target, statistic, "RQMC"), [])
        if not mc or not rqmc:
            continue
        mc_rmse = float(np.sqrt(np.mean(np.square(mc))))
        rqmc_rmse = float(np.sqrt(np.mean(np.square(rqmc))))
        gain = math.inf if rqmc_rmse == 0.0 else mc_rmse / rqmc_rmse
        result.append(
            {
                "budget": budget,
                "target": target,
                "statistic": statistic,
                "repetitions": min(len(mc), len(rqmc)),
                "mc_rmse": mc_rmse,
                "rqmc_rmse": rqmc_rmse,
                "gain_G": gain,
            }
        )
    return result


def _run_main_variant(
    *,
    config: RQ1Config,
    variant: _MainVariant,
    surrogate: Release25Surrogate,
    output_root: Path,
) -> dict[str, object]:
    variant_root = output_root / variant.key
    fields_root = variant_root / "fields"
    figures_root = variant_root / "figures"
    scratch_root = variant_root / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)

    sampler = DiagnosticPermeabilitySampler(
        sampler=variant.sampler,
        diagnostics=variant.diagnostics,
    )
    qoi_accumulator = RQ1QoIAccumulator(
        background_temperature=config.background_temperature,
        receptors=config.receptors,
        mean_anomaly_roi=config.mean_anomaly_roi,
    )
    store_accumulator = DiskTemperatureStoreAccumulator(
        scratch_root / "temperature_samples.npy",
        expected_samples=config.max_budget,
    )
    result = MonteCarloRunner(sampler=sampler, surrogate=surrogate).run(
        n_samples=config.max_budget,
        ddof=1,
        accumulators=(qoi_accumulator, store_accumulator),
    )
    qoi_samples = result.accumulator_results["rq1_qoi"]
    store = result.accumulator_results["rq1_temperature_store"]
    diagnostic_result = variant.diagnostics.finalize()

    input_figures = plot_permeability_diagnostics(
        diagnostic_result,
        figures_root / "input",
        cell_size_m=config.cell_size_m,
        observation_indices=(
            np.asarray([[row, col] for row, col, _ in config.observations], dtype=int)
            if variant.key == "C_conditional_grf_mc"
            else None
        ),
    )

    field_paths = write_temperature_field_products(
        temperature_store=store.path,
        final_mean=result.mean,
        final_std=result.std,
        output_directory=fields_root,
        quantiles=config.quantiles,
        chunk_rows=config.quantile_chunk_rows,
    )
    temperature_figures = plot_rq1_temperature_fields(
        fields_root,
        figures_root / "temperature",
        cell_size_m=config.cell_size_m,
        title_prefix=variant.key,
    )
    qoi_figures = plot_rq1_qoi_distributions(
        mean_anomaly=qoi_samples.mean_anomaly,
        receptor_values=qoi_samples.receptor_values,
        receptor_indices=qoi_samples.receptor_indices,
        directory=figures_root / "qoi",
        title_prefix=variant.key,
    )

    global_metrics = global_uncertainty_metrics(result.std)
    qoi_reference = _qoi_reference(qoi_samples)
    field_points = prefix_field_convergence(
        temperature_store=store.path,
        checkpoints=config.budgets,
        reference_mean=result.mean,
        reference_std=result.std,
        ddof=1,
        chunk_rows=config.quantile_chunk_rows,
    )
    convergence_rows = _main_convergence_rows(
        variant=variant.key,
        method=variant.method,
        field_points=field_points,
        global_reference=global_metrics,
        qoi_samples=qoi_samples,
        qoi_reference=qoi_reference,
        checkpoints=config.budgets,
    )

    input_payload = _diagnostics_payload(diagnostic_result, config)
    if (
        variant.key == "C_conditional_grf_mc"
        and config.conditioning_tolerance_log10 is not None
        and input_payload["conditioning_within_tolerance"] is False
    ):
        raise RuntimeError(
            "conditional GRF samples violate configured conditioning_tolerance_log10"
        )

    if not config.retain_scratch:
        try:
            Path(store.path).unlink()
            scratch_root.rmdir()
        except OSError:
            pass

    return {
        "result": result,
        "qoi_samples": qoi_samples,
        "qoi_reference": qoi_reference,
        "global_metrics": global_metrics,
        "input_diagnostics": input_payload,
        "convergence_rows": convergence_rows,
        "qoi_rows": _qoi_rows(
            experiment_id=config.experiment_id,
            variant=variant.key,
            method=variant.method,
            repetition="main",
            samples=qoi_samples,
        ),
        "receptor_rows": _receptor_metric_rows(
            variant=variant.key,
            method=variant.method,
            repetition="main",
            budget=config.max_budget,
            samples=qoi_samples,
        ),
        "field_paths": {key: str(path) for key, path in field_paths.items()},
        "input_figures": {key: str(path) for key, path in input_figures.items()},
        "temperature_figures": {
            key: str(path) for key, path in temperature_figures.items()
        },
        "qoi_figures": {key: str(path) for key, path in qoi_figures.items()},
    }


def run_rq1(config: RQ1Config) -> dict[str, Path]:
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_yaml(output_root / "config.yaml", config.to_dict())

    runtime = Release25Runtime.from_paths(
        release25_repo=config.release25_repo,
        cnn1_dir=config.cnn1_dir,
        cnn2_dir=config.cnn2_dir,
        prepared_pki_dir=config.prepared_pki_dir,
        run_id=config.fixed_run_id,
        device=config.device,
        random_k=config.random_k,
        streamline_method=config.streamline_method,
    )
    configure_release25_streamlines(
        runtime.adapter,
        mode=config.streamline_mode,
        max_nfev=config.streamline_max_nfev,
        diagnostics=config.streamline_diagnostics,
        slow_streamline_seconds=config.streamline_slow_seconds,
    )
    surrogate = Release25Surrogate(
        adapter=runtime.adapter,
        fixed_inputs=runtime.scenario.fixed,
        device=config.device,
    )

    shape = tuple(int(v) for v in runtime.scenario.shape)
    prior, conditional = _build_grf_maps(config, shape)
    domain_size_m = (shape[0] * config.cell_size_m, shape[1] * config.cell_size_m)

    perlin_map = PerlinCoordinatePermeabilityMap(
        shape=shape,
        domain_size_m=domain_size_m,
        frequency=RELEASE25_PERLIN_FREQUENCY,
        k_min=RELEASE25_PERLIN_K_MIN,
        k_max=RELEASE25_PERLIN_K_MAX,
    )
    perlin = UniformCoordinatePermeabilitySampler(
        field_map=perlin_map,
        n_samples=config.max_budget,
        batch_size=config.batch_size,
        seed=config.main_seed,
    )
    unconditional_sampler = GaussianCoordinatePermeabilitySampler(
        field_map=prior,
        n_samples=config.max_budget,
        batch_size=config.batch_size,
        seed=config.main_seed,
    )
    conditional_sampler = GaussianCoordinatePermeabilitySampler(
        field_map=conditional,
        n_samples=config.max_budget,
        batch_size=config.batch_size,
        seed=config.main_seed,
    )
    variants = (
        _MainVariant(
            "A_perlin_mc",
            "MC",
            perlin,
            _diagnostics(config, conditional=False),
        ),
        _MainVariant(
            "B_unconditional_grf_mc",
            "MC",
            unconditional_sampler,
            _diagnostics(config, conditional=False),
        ),
        _MainVariant(
            "C_conditional_grf_mc",
            "MC",
            conditional_sampler,
            _diagnostics(config, conditional=True),
        ),
    )

    main_results: dict[str, dict[str, object]] = {}
    convergence_rows: list[dict[str, object]] = []
    qoi_rows: list[dict[str, object]] = []
    receptor_rows: list[dict[str, object]] = []
    input_diagnostics: dict[str, object] = {}
    global_metrics: dict[str, object] = {}

    for variant in variants:
        completed = _run_main_variant(
            config=config,
            variant=variant,
            surrogate=surrogate,
            output_root=output_root,
        )
        main_results[variant.key] = completed
        convergence_rows.extend(completed["convergence_rows"])
        qoi_rows.extend(completed["qoi_rows"])
        receptor_rows.extend(completed["receptor_rows"])
        input_diagnostics[variant.key] = completed["input_diagnostics"]
        global_metrics[variant.key] = {
            **completed["global_metrics"],
            "sample_count": config.max_budget,
            "qoi": completed["qoi_reference"],
        }

    reference = main_results["C_conditional_grf_mc"]
    reference_result = reference["result"]
    reference_global = reference["global_metrics"]
    reference_qoi = reference["qoi_reference"]

    repeated_rows: list[dict[str, object]] = []
    for repetition, seed in enumerate(config.repetition_seeds, start=1):
        for method in ("MC", "RQMC"):
            if method == "MC":
                sampler = GaussianCoordinatePermeabilitySampler(
                    field_map=conditional,
                    n_samples=config.max_comparison_budget,
                    batch_size=config.batch_size,
                    seed=seed,
                )
                variant_key = "C_conditional_grf_mc"
            else:
                sampler = ScrambledSobolGaussianPermeabilitySampler(
                    field_map=conditional,
                    n_samples=config.max_comparison_budget,
                    batch_size=config.batch_size,
                    seed=seed,
                )
                variant_key = "D_conditional_grf_rqmc"

            field_accumulator = ReferenceConvergenceFieldStatisticsAccumulator(
                checkpoints=config.comparison_budgets,
                reference_mean=reference_result.mean,
                reference_std=reference_result.std,
                ddof=1,
            )
            qoi_accumulator = RQ1QoIAccumulator(
                background_temperature=config.background_temperature,
                receptors=config.receptors,
                mean_anomaly_roi=config.mean_anomaly_roi,
            )
            repeated_result = MonteCarloRunner(
                sampler=sampler,
                surrogate=surrogate,
            ).run(
                n_samples=config.max_comparison_budget,
                ddof=1,
                accumulators=(qoi_accumulator,),
                field_statistics_accumulator=field_accumulator,
            )
            repeated_qoi = repeated_result.accumulator_results["rq1_qoi"]
            rows = _repeated_convergence_rows(
                variant=variant_key,
                method=method,
                repetition=repetition,
                field_accumulator=field_accumulator,
                global_reference=reference_global,
                qoi_samples=repeated_qoi,
                qoi_reference=reference_qoi,
                checkpoints=config.comparison_budgets,
            )
            repeated_rows.extend(rows)
            convergence_rows.extend(rows)
            qoi_rows.extend(
                _qoi_rows(
                    experiment_id=config.experiment_id,
                    variant=variant_key,
                    method=method,
                    repetition=repetition,
                    samples=repeated_qoi,
                )
            )
            for budget in config.comparison_budgets:
                receptor_rows.extend(
                    _receptor_metric_rows(
                        variant=variant_key,
                        method=method,
                        repetition=repetition,
                        budget=budget,
                        samples=repeated_qoi,
                    )
                )

    comparison_rows = _aggregate_mc_rqmc(repeated_rows)
    global_metrics["D_conditional_grf_rqmc"] = {
        "role": "randomized_QMC_efficiency_comparison",
        "max_budget": config.max_comparison_budget,
        "repetitions": config.repetitions,
        "reference": "C_conditional_grf_mc empirical MC at max main budget",
    }

    artifacts: dict[str, Path] = {}
    artifacts["input_diagnostics"] = write_json(
        output_root / "input_diagnostics.json", input_diagnostics
    )
    artifacts["global_metrics"] = write_json(
        output_root / "global_metrics.json", global_metrics
    )
    artifacts["convergence"] = write_csv(
        output_root / "convergence.csv", convergence_rows
    )
    artifacts["receptor_metrics"] = write_csv(
        output_root / "receptor_metrics.csv", receptor_rows
    )
    artifacts["qoi_samples"] = write_csv(
        output_root / "qoi_samples.csv", qoi_rows
    )
    artifacts["mc_rqmc_comparison"] = write_csv(
        output_root / "mc_rqmc_comparison.csv", comparison_rows
    )

    figures_root = output_root / "figures"
    plot_rq1_convergence(convergence_rows, figures_root / "convergence")
    plot_mc_rqmc_comparison(comparison_rows, figures_root / "mc_vs_rqmc")

    project_root = Path(__file__).resolve().parents[3]
    metadata = {
        "experiment_id": config.experiment_id,
        "research_question": RQ1_WORDING,
        "scope": (
            "Permeability-input uncertainty only; pressure and heat-pump inputs are fixed "
            "and the release25 LGCNN weights are frozen."
        ),
        "variants": [
            "A_perlin_mc",
            "B_unconditional_grf_mc",
            "C_conditional_grf_mc",
            "D_conditional_grf_rqmc",
        ],
        "empirical_reference": {
            "variant": "C_conditional_grf_mc",
            "sample_count": config.max_budget,
            "description": "largest conditional-GRF iid-MC run; empirical reference, not ground truth",
        },
        "sampling_laws": {
            "A_perlin_mc": perlin.metadata,
            "B_unconditional_grf_mc": unconditional_sampler.metadata,
            "C_conditional_grf_mc": conditional_sampler.metadata,
            "D_conditional_grf_rqmc": {
                "sampler": "ScrambledSobolGaussianPermeabilitySampler",
                "scramble": True,
                "coordinate_distribution": "iid_standard_normal_via_inverse_cdf",
                "uniform_design": "scipy.stats.qmc.Sobol(scramble=True)",
                "coordinate_dimension": int(conditional.dimension),
                "field_map": conditional.metadata,
                "comparison_budgets": list(config.comparison_budgets),
                "repetition_seeds": list(config.repetition_seeds),
            },
        },
        "rqmc": {
            "construction": "scrambled Sobol U(0,1)^d followed by component-wise standard-normal inverse CDF",
            "comparison_budgets": list(config.comparison_budgets),
            "repetitions": config.repetitions,
            "seeds": list(config.repetition_seeds),
            "gain_definition": "G(N)=RMSE_MC(N)/RMSE_RQMC(N)",
        },
        "streamlines": {
            "mode": config.streamline_mode,
            "method": config.streamline_method,
            "max_nfev": config.streamline_max_nfev,
            "diagnostics": config.streamline_diagnostics,
            "slow_seconds": config.streamline_slow_seconds,
        },
        "field_quantiles": {
            "probabilities": list(config.quantiles),
            "method": "numpy empirical quantile, linear interpolation",
            "storage": "disk-backed exact temperature samples with spatial chunking",
        },
        "excluded_from_rq1": [
            "threshold/risk and plume metrics (RQ2)",
            "sensitivity analysis (RQ3)",
            "PCE",
            "LGCNN-PFLOTRAN/model-discrepancy uncertainty (RQ6)",
        ],
        "fixed_run_id": config.fixed_run_id,
        "background_temperature_c": config.background_temperature,
        "receptors": [list(v) for v in config.receptors],
        "mean_anomaly_roi": (
            None if config.mean_anomaly_roi is None else list(config.mean_anomaly_roi)
        ),
        "budgets": list(config.budgets),
        "runtime": runtime_versions(),
        "provenance": {
            "masterthesis_git_sha": git_head(project_root),
            "release25_git_sha": git_head(config.release25_repo),
            "cnn1_checkpoint_sha256": sha256_file(runtime.adapter.cnn1.checkpoint),
            "cnn3_checkpoint_sha256": sha256_file(runtime.adapter.cnn2.checkpoint),
            "reference_published_model_doi": "10.18419/DARUS-5080",
            "reference_dataset": "dataset_giant_100hp_varyK",
        },
    }
    artifacts["metadata"] = write_json(output_root / "metadata.json", metadata)

    return artifacts
