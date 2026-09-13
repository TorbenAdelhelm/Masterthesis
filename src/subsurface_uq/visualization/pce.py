from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Array = np.ndarray


def _load_required_array(archive: np.lib.npyio.NpzFile, name: str) -> Array:
    if name not in archive.files:
        raise ValueError(f"PCE archive is missing required array {name!r}")
    value = np.asarray(archive[name])
    if not np.all(np.isfinite(value)):
        raise ValueError(f"PCE archive array {name!r} contains non-finite values")
    return value


def _term_label(alpha: Array) -> str:
    powers = np.asarray(alpha, dtype=int).ravel()
    factors: list[str] = []
    for index, power in enumerate(powers, start=1):
        if power == 1:
            factors.append(rf"\xi_{index}")
        elif power > 1:
            factors.append(rf"\xi_{index}^{{{power}}}")
    return r"$1$" if not factors else "$" + " ".join(factors) + "$"


def plot_pce_archive(
    archive_path: str | Path,
    output_dir: str | Path,
    *,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Create a compact diagnostic plot set from one saved PCE result archive.

    The current proof-of-concept intentionally keeps plotting small: parity and
    residual plots assess validation accuracy, while a coefficient plot exposes
    the fitted polynomial structure. Cross-run convergence plots are deferred
    until the experimental sweep format is fixed.
    """

    source = Path(archive_path).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or source.stem

    with np.load(source, allow_pickle=False) as archive:
        truth = _load_required_array(archive, "validation_qoi").astype(np.float64)
        prediction = _load_required_array(
            archive, "validation_prediction"
        ).astype(np.float64)
        multi_indices = _load_required_array(archive, "multi_indices").astype(int)
        coefficients = _load_required_array(archive, "coefficients").astype(np.float64)

    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError(
            "validation_qoi and validation_prediction must have matching shape [N]"
        )
    if truth.size == 0:
        raise ValueError("PCE archive contains no validation samples")
    if multi_indices.ndim != 2:
        raise ValueError("multi_indices must have shape [P, dimension]")
    if coefficients.ndim != 1 or coefficients.shape[0] != multi_indices.shape[0]:
        raise ValueError("coefficients must have one value per multi-index")

    paths: dict[str, Path] = {}

    # 1) Validation parity plot.
    lower = float(min(np.min(truth), np.min(prediction)))
    upper = float(max(np.max(truth), np.max(prediction)))
    if upper <= lower:
        padding = max(abs(lower), 1.0) * 0.05
        lower -= padding
        upper += padding

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(truth, prediction)
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel("LGCNN validation QoI")
    ax.set_ylabel("PCE-predicted QoI")
    ax.set_title("PCE validation parity")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = target_dir / f"{stem}_parity.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["parity"] = path

    # 2) Residuals in validation-sample order.
    residual = prediction - truth
    sample_index = np.arange(1, truth.size + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(sample_index, residual, marker="o")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Validation sample")
    ax.set_ylabel("PCE - LGCNN QoI")
    ax.set_title("PCE validation residuals")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = target_dir / f"{stem}_residuals.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["residuals"] = path

    # 3) Fitted PCE coefficients. The current 2-D proof-of-concept has a small
    # basis, so displaying every term is clearer than introducing ranking logic.
    labels = [_term_label(alpha) for alpha in multi_indices]
    width = max(6.0, min(12.0, 0.45 * len(labels)))
    fig, ax = plt.subplots(figsize=(width, 4.2))
    positions = np.arange(len(coefficients))
    ax.bar(positions, coefficients)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("PCE basis term")
    ax.set_ylabel("Coefficient")
    ax.set_title("Fitted PCE coefficients")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = target_dir / f"{stem}_coefficients.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["coefficients"] = path

    return paths
