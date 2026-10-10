"""Generate and compare explicit reference-centered uncertainty scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..sampling.darus_real import load_release25_raw_permeability_run
from ..sampling.coordinates import GaussianCoordinatePermeabilitySampler
from ..sampling.geostatistical_target import (
    Base10LognormalTarget,
    HydraulicConductivityValidationAccumulator,
    validate_hydraulic_conductivity_marginal,
)
from ..sampling.hydraulic_conductivity import (
    FluidProperties,
    hydraulic_conversion_metadata,
    permeability_to_hydraulic_conductivity,
)
from ..sampling.reference_field import save_reference_field_input_model
from ..sampling.spatial_diagnostics import compare_reference_ensemble


def coarsen_reference(reference, factor):
    """Geometric block means, at actual block centers (no decimation offset)."""
    reference = np.asarray(reference)
    if (reference.ndim != 2 or factor <= 0 or any(n % factor for n in reference.shape)
            or not np.all(np.isfinite(reference)) or np.any(reference <= 0)):
        raise ValueError("positive reference dimensions must be divisible by coarsen-factor")
    height, width = reference.shape
    logs = np.log10(reference.astype(float)).reshape(height//factor, factor, width//factor, factor)
    return np.power(10., logs.mean(axis=(1, 3)))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset-root", help="Original DaRUS dataset; requires --reference-run")
    sources.add_argument("--reference-npy", help="One physical 2-D permeability field in m²")
    parser.add_argument("--reference-run")
    parser.add_argument("--cell-size-m", type=float, default=5.)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("ROW_START", "ROW_STOP", "COL_START", "COL_STOP"))
    parser.add_argument("--coarsen-factor", type=int, default=1)
    parser.add_argument("--diagnostic-factor", type=int, default=1,
                        help="Geometric block means for bounded-memory matched-support diagnostics")
    parser.add_argument("--residual-std-log10-k", type=float, required=True,
                        help="Explicit residual amplitude around the heterogeneous reference; not the pooled field std")
    parser.add_argument("--length-scale-row-m", "--length-scale-y-m", dest="length_scale_row_m",
                        type=float, required=True, help="Correlation length along array rows")
    parser.add_argument("--length-scale-col-m", "--length-scale-x-m", dest="length_scale_col_m",
                        type=float, required=True, help="Correlation length along array columns")
    parser.add_argument("--covariance-model", choices=("matern32", "exponential"), default="matern32")
    truncation = parser.add_mutually_exclusive_group()
    truncation.add_argument("--energy-thresholds", type=float, nargs="+", default=None)
    truncation.add_argument("--n-modes", type=int, nargs="+", help="Explicit representation sensitivity dimensions")
    parser.add_argument("--center", choices=("median", "arithmetic-mean"), default="median")
    parser.add_argument("--observations", help="YAML with local yx_m, log10_permeability_m2 and std_log10 arrays")
    parser.add_argument("--permeability-convention", choices=("historical-training", "physical"), required=True,
                        help="Convention already used by reference and observation values; no implicit conversion")
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--mu", type=float, default=1.0e-3)
    parser.add_argument("--g", type=float, default=9.80665)
    parser.add_argument("--target-log10-mean", type=float, default=-3.0)
    spread = parser.add_mutually_exclusive_group()
    spread.add_argument("--target-log10-std", type=float, default=None,
                        help="Base-10 log-space standard deviation; teaching example uses 0.5")
    spread.add_argument("--target-log10-variance", type=float, default=None,
                        help="Explicit base-10 log-space variance; use only if variance, not sigma, is intended")
    parser.add_argument("--target-lb-m-s", type=float, default=1.0e-4)
    parser.add_argument("--target-ub-m-s", type=float, default=5.0e-2)
    parser.add_argument("--minimum-interval-fraction", type=float, default=0.95)
    parser.add_argument("--max-abs-log10-mean-error", type=float)
    parser.add_argument("--max-abs-log10-std-error", type=float)
    parser.add_argument("--max-qq-rmse-standardized", type=float)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4901)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset_root:
        if not args.reference_run:
            parser.error("--dataset-root requires an explicit --reference-run")
        if args.permeability_convention != "historical-training":
            parser.error("original DaRUS RUN inputs use the historical-training convention")
        reference = load_release25_raw_permeability_run(args.dataset_root, args.reference_run,
                                                       cell_size_m=args.cell_size_m)
    else:
        reference = np.load(args.reference_npy, allow_pickle=False)
    if reference.ndim != 2:
        parser.error("reference must be one 2-D field")
    if args.roi:
        r0, r1, c0, c1 = args.roi
        if not (0 <= r0 < r1 <= reference.shape[0] and 0 <= c0 < c1 <= reference.shape[1]):
            parser.error("--roi must be a non-empty half-open rectangle within the reference")
        reference = reference[r0:r1, c0:c1]
    reference = coarsen_reference(reference, args.coarsen_factor)
    diagnostic_reference = coarsen_reference(reference, args.diagnostic_factor)
    cell_size = args.cell_size_m * args.coarsen_factor
    conditioning = {}
    if args.observations:
        observations = yaml.safe_load(Path(args.observations).read_text(encoding="utf-8"))
        if observations.get("permeability_convention") != args.permeability_convention:
            parser.error("observation and reference permeability conventions must match explicitly")
        conditioning = {
            "observation_coordinates_yx_m": observations["yx_m"],
            "observation_log10_k": observations["log10_permeability_m2"],
            "observation_std_log10_k": observations["std_log10"],
        }
    target_payload = {
        "log10_mean": args.target_log10_mean,
        "interval_m_s": [args.target_lb_m_s, args.target_ub_m_s],
        "minimum_interval_fraction": args.minimum_interval_fraction,
    }
    if args.target_log10_variance is not None:
        target_payload["log10_variance"] = args.target_log10_variance
    else:
        target_payload["log10_std"] = 0.5 if args.target_log10_std is None else args.target_log10_std
    target = Base10LognormalTarget.from_mapping(target_payload)
    tolerances = {
        key: value for key, value in {
            "max_abs_log10_mean_error": args.max_abs_log10_mean_error,
            "max_abs_log10_std_error": args.max_abs_log10_std_error,
            "max_qq_rmse_standardized": args.max_qq_rmse_standardized,
        }.items() if value is not None
    } or None
    fluid = FluidProperties(args.rho, args.mu, args.g)
    conversion = hydraulic_conversion_metadata(args.permeability_convention, fluid=fluid)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = {
        "dataset_root": str(Path(args.dataset_root).resolve()) if args.dataset_root else None,
        "reference_run": args.reference_run, "reference_npy": args.reference_npy,
        "permeability_convention": args.permeability_convention, "roi": args.roi,
        "coarsen_factor": args.coarsen_factor, "coarsening": "geometric_block_mean",
        "original_cell_size_m": args.cell_size_m, "generated_cell_size_m": cell_size,
        "diagnostic_factor": args.diagnostic_factor,
        "hydraulic_conversion": conversion,
        "distribution_target": target.metadata,
        "uncertainty_status": "explicit sensitivity assumption; not calibrated interpolation error",
    }
    summary = {"input_law": "reference-centered-lognormal", "source": source,
               "n_samples": args.n_samples, "seed": args.seed,
               "distribution_target": target.metadata,
               "truncation_sensitivity": []}
    choices = ([(None, n) for n in args.n_modes] if args.n_modes
               else [(e, None) for e in (args.energy_thresholds or [.95, .99, .999])])
    from ..sampling.scenarios import ReferenceScenario
    for energy, n_modes in choices:
        scenario = ReferenceScenario(reference_run=args.reference_run or "explicit-npy-reference",
                                     sigma_R=args.residual_std_log10_k, ell_y=args.length_scale_row_m,
                                     ell_x=args.length_scale_col_m, covariance_family=args.covariance_model,
                                     energy_threshold=energy, n_modes=n_modes, center=args.center)
        unconditional, conditional = scenario.build_maps(reference, cell_size_m=cell_size, **conditioning)
        directory = root / (f"energy_{energy:g}" if energy is not None else f"modes_{n_modes}")
        directory.mkdir(exist_ok=True)
        save_reference_field_input_model(directory / "stochastic_input_model.yaml", unconditional,
                                         source_metadata=source, **conditioning)
        sampler = GaussianCoordinatePermeabilitySampler(field_map=conditional, n_samples=args.n_samples,
                                                        batch_size=args.batch_size, seed=args.seed)
        ensemble = np.lib.format.open_memmap(directory / "generated_fields.npy", mode="w+", dtype=np.float32,
                                            shape=(args.n_samples, *reference.shape))
        diagnostic_fields = []
        distribution = HydraulicConductivityValidationAccumulator(target, tolerances=tolerances)
        sum_squared_residual = 0.
        start = 0
        for batch in sampler:
            ensemble[start:start + len(batch)] = batch
            residual = np.log10(batch.astype(float)) - np.log10(reference)
            sum_squared_residual += float(np.sum(residual**2))
            for field in batch:
                kh = permeability_to_hydraulic_conductivity(
                    field, convention=args.permeability_convention, fluid=fluid
                )
                distribution.update(kh)
                diagnostic_fields.append(coarsen_reference(field, args.diagnostic_factor))
            start += len(batch)
        ensemble.flush()
        fidelity = compare_reference_ensemble(diagnostic_reference, np.asarray(diagnostic_fields),
                                              cell_size_m=cell_size * args.diagnostic_factor)
        (directory / "fidelity.json").write_text(json.dumps(fidelity, indent=2, allow_nan=False), encoding="utf-8")
        distribution_report = distribution.finalize()
        reference_kh = permeability_to_hydraulic_conductivity(
            reference, convention=args.permeability_convention, fluid=fluid
        )
        distribution_report.update({
            "reference_marginal": validate_hydraulic_conductivity_marginal(
                reference_kh, target, tolerances=tolerances
            ),
            "residual_law": {
                "sigma_R_log10": args.residual_std_log10_k,
                "interpretation": "Residual spread around the reference; not pooled full-field sigma_log10.",
            },
            "hydraulic_conversion": conversion,
            "spatial_reference_fidelity": fidelity,
            "sample_filtering": None,
        })
        (directory / "distribution_validation.json").write_text(
            json.dumps(distribution_report, indent=2, allow_nan=False), encoding="utf-8"
        )
        variance = unconditional.prior.pointwise_log10_variance()
        row = {"scenario": scenario.manifest(), "n_modes_requested": n_modes,
               "energy_requested": energy, "energy_retained": unconditional.prior.retained_energy_fraction,
               "dimension": conditional.dimension, "prior_dimension": unconditional.dimension,
               "mean_retained_pointwise_variance": float(variance.mean()),
               "min_retained_pointwise_variance": float(variance.min()),
               "max_retained_pointwise_variance": float(variance.max()),
               "sample_reference_difference_rms_log10": float(np.sqrt(sum_squared_residual / ensemble.size)),
               "generated_full_field_log10_std": distribution_report["pooled"]["log10_std"],
               "generated_full_field_log10_std_error": distribution_report["pooled"]["log10_std_error"],
               "generated_interval_fraction": distribution_report["pooled"]["fraction_inside_interval"],
               "marginal_wasserstein_log10": fidelity["marginal"]["wasserstein_distance_log10"],
               "radial_power_relative_l2": fidelity["spectra"]["radial_power_relative_l2"],
               "angular_power_relative_l2": fidelity["spectra"]["angular_power_relative_l2"],
               "reference_axis_power_fraction": fidelity["spectra"]["reference"]["axis_band_power_fraction"],
               "generated_axis_power_fraction": fidelity["spectra"]["generated"]["axis_band_power_fraction"]}
        summary["truncation_sensitivity"].append(row)
        print(f"{directory.name}: dimension={conditional.dimension}, retained={row['energy_retained']:.6f}")
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved reference scenario generator and comparisons to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
