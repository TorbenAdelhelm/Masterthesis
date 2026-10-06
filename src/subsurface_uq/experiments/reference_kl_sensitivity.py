"""Paired Gaussian-coordinate truncation study of downstream temperature QoIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from ..io import sha256_file
from ..sampling.input_model import load_conditional_kl_input_model


def compare_qoi_truncations(maps, evaluate_field, *, n_samples, seed=4901, method="MC"):
    """evaluate_field maps a physical field to a fixed vector of temperature QoIs.

    Maps must differ only in truncation. Common coordinates pair identical KL
    eigenmodes. The largest dimension is a comparison reference, not truth.
    """
    if len(maps) < 2 or n_samples < 2:
        raise ValueError("at least two maps and two samples required")
    def signature(field_map):
        metadata = field_map.metadata
        # Compare scientific inputs directly, ignoring truncation-dependent metadata.
        return (metadata["reference_array_sha256"], field_map.field_shape, field_map.domain_size_m,
                field_map.center, field_map.prior.std_log10_k, field_map.prior.length_scale_m,
                field_map.prior.covariance_model)
    if any(signature(m) != signature(maps[0]) or m.gaussian_map is not m.prior for m in maps):
        raise ValueError("QoI study requires unconditional maps differing only in truncation")
    dimension = max(m.dimension for m in maps)
    if method == "MC":
        xi = np.random.default_rng(seed).normal(size=(n_samples, dimension))
    elif method == "RQMC":
        if n_samples & (n_samples - 1):
            raise ValueError("RQMC budget must be a power of two")
        uniforms = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(int(np.log2(n_samples)))
        xi = ndtri(np.clip(uniforms, np.nextafter(0., 1.), np.nextafter(1., 0.)))
    else:
        raise ValueError("method must be MC or RQMC")
    outputs = []
    for field_map in maps:
        rows = [np.atleast_1d(evaluate_field(field_map.map_coordinates(row[:field_map.dimension]))) for row in xi]
        values = np.asarray(rows, dtype=float)
        if values.ndim != 2 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("temperature QoI evaluator must return finite scalar/vector values")
        outputs.append(values)
    if len({v.shape for v in outputs}) != 1:
        raise ValueError("all truncations must evaluate the same QoIs")
    reference_index = max(range(len(maps)), key=lambda i: maps[i].dimension)
    reference = outputs[reference_index]
    rows = []
    for field_map, values in zip(maps, outputs):
        difference = values - reference
        rows.append({"dimension": field_map.dimension,
                     "retained_energy": field_map.prior.retained_energy_fraction,
                     "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0, ddof=1).tolist(),
                     "quantiles_05_50_95": np.quantile(values, [.05, .5, .95], axis=0).tolist(),
                     "paired_rmse_to_largest": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
                     "mean_difference_to_largest": difference.mean(axis=0).tolist(),
                     "std_difference_to_largest": (values.std(axis=0, ddof=1) - reference.std(axis=0, ddof=1)).tolist()})
    return {"method": method, "seed": seed, "n_samples": n_samples,
            "reference_index": reference_index, "truncations": rows,
            "interpretation": "representation sensitivity under one assumed law; largest dimension is not truth; repeat seeds/budgets to assess integration uncertainty"}, np.stack(outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-models", nargs="+", required=True)
    parser.add_argument("--rq1-config", required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=4901)
    parser.add_argument("--method", choices=("MC", "RQMC"), default="MC")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    from ..rq1.config import load_rq1_config
    from ..surrogates import Release25Surrogate
    from ..surrogates.release25_runtime import Release25Runtime
    from ..surrogates.bounded_streamlines import configure_release25_streamlines
    config = load_rq1_config(args.rq1_config)
    loaded = [load_conditional_kl_input_model(path) for path in args.input_models]
    # Do not silently discard conditioning from an artifact.
    if any(model.payload.get("conditioning") for model in loaded):
        parser.error("paired KL study currently requires unconditioned reference artifacts")
    maps = [model.unconditional for model in loaded]
    runtime = Release25Runtime.from_paths(
        release25_repo=config.release25_repo, cnn1_dir=config.cnn1_dir, cnn2_dir=config.cnn2_dir,
        prepared_pki_dir=config.prepared_pki_dir, run_id=config.fixed_run_id, device=config.device,
        random_k=config.random_k, streamline_method=config.streamline_method)
    for field_map in maps:
        if (field_map.field_shape != tuple(runtime.scenario.shape)
                or not np.allclose(field_map.domain_size_m, np.asarray(runtime.scenario.shape) * config.cell_size_m)):
            parser.error("input maps must match the fixed pretrained-model grid and cell size")
    configure_release25_streamlines(runtime.adapter, mode=config.streamline_mode,
                                    max_nfev=config.streamline_max_nfev, diagnostics=config.streamline_diagnostics,
                                    slow_streamline_seconds=config.streamline_slow_seconds)
    surrogate = Release25Surrogate(runtime.adapter, runtime.scenario.fixed, device=config.device)
    def evaluate(field):
        temperature = surrogate.predict_temperature(field)
        values = [float(temperature[row, col]) for row, col in config.receptors]
        if config.mean_anomaly_roi is not None:
            r0, r1, c0, c1 = config.mean_anomaly_roi
            if not (0 <= r0 < r1 <= temperature.shape[0] and 0 <= c0 < c1 <= temperature.shape[1]):
                raise ValueError("temperature QoI ROI outside output grid")
            values.append(float(temperature[r0:r1, c0:c1].mean() - config.background_temperature))
        return values
    report, values = compare_qoi_truncations(maps, evaluate, n_samples=args.n_samples, seed=args.seed, method=args.method)
    report.update(input_models=[{"path": str(m.source_path), "sha256": sha256_file(m.source_path)} for m in loaded],
                  rq1_config=config.to_dict(), qoi_names=[f"temperature_{r}_{c}" for r, c in config.receptors]
                  + (["mean_temperature_anomaly"] if config.mean_anomaly_roi else []))
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "paired_temperature_qoi.npy", values)
    (root / "qoi_truncation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
