from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import importlib
import sys

import torch
import yaml

Tensor = torch.Tensor


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def _one_file(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_file():
        return direct.resolve()
    matches = sorted(path.resolve() for path in root.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"could not find {name!r} below {root}")
    if len(matches) > 1:
        listing = "\n".join(f"  - {path}" for path in matches)
        raise ValueError(f"multiple {name!r} files found below {root}:\n{listing}")
    return matches[0]


def _checkpoint(root: Path) -> Path:
    try:
        return _one_file(root, "model.pt")
    except FileNotFoundError:
        matches = sorted(path.resolve() for path in root.rglob("*.pt") if path.is_file())
        if not matches:
            raise FileNotFoundError(f"no PyTorch checkpoint found below {root}")
        if len(matches) > 1:
            listing = "\n".join(f"  - {path}" for path in matches)
            raise ValueError(
                "no model.pt was found and more than one .pt checkpoint exists; "
                f"the standard DaRUS model archive is expected:\n{listing}"
            )
        return matches[0]


def _hps_value(hps: Mapping[str, Any], key: str) -> Any:
    if key not in hps:
        raise KeyError(f"hyperparameter {key!r} missing from HPS_options.yaml")
    value = hps[key]
    if isinstance(value, Mapping) and "values" in value:
        values = value["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"HPS entry {key!r} contains no values")
        # This is the rule used by release25's non-Optuna training path.
        return values[0]
    return value


def _release25_code_dir(release25_repo: str | Path) -> Path:
    root = Path(release25_repo).expanduser().resolve()
    if (root / "processing" / "networks" / "unetVariants.py").is_file():
        return root
    code = root / "code"
    if (code / "processing" / "networks" / "unetVariants.py").is_file():
        return code
    raise FileNotFoundError(
        "release25 code not found. Pass either the Heat-Plume-Prediction release25 "
        "repository root or its code/ directory."
    )


def _install_release25_import_path(release25_repo: str | Path) -> Path:
    code = _release25_code_dir(release25_repo)
    value = str(code)
    if value not in sys.path:
        sys.path.insert(0, value)
    return code


@dataclass(frozen=True)
class Release25Normalizer:
    """Reproduce the normalization stored in a release25 ``info.yaml`` file."""

    info: Mapping[str, Any]
    out_min: float = 0.0
    out_max: float = 1.0

    def _stats(self, data_type: str) -> dict[int, Mapping[str, Any]]:
        section = self.info.get(data_type)
        if not isinstance(section, Mapping):
            raise KeyError(f"{data_type!r} missing from info metadata")
        result: dict[int, Mapping[str, Any]] = {}
        for name, stats in section.items():
            if not isinstance(stats, Mapping) or "index" not in stats:
                raise ValueError(f"invalid normalization metadata for {name!r}")
            result[int(stats["index"])] = stats
        return result

    @staticmethod
    def _channel_dim(data: Tensor) -> int:
        if data.ndim == 3:
            return 0
        if data.ndim == 4:
            return 1
        raise ValueError(f"expected [C,H,W] or [B,C,H,W], got {tuple(data.shape)}")

    def _apply(self, x: Tensor, stats: Mapping[str, Any], reverse: bool) -> Tensor:
        norm = stats.get("norm")
        if norm is None:
            return x
        if norm == "Rescale":
            minimum = float(stats["min"])
            maximum = float(stats["max"])
            delta = maximum - minimum
            if delta == 0.0:
                raise ValueError("cannot rescale a constant channel")
            if reverse:
                return ((x - self.out_min) / (self.out_max - self.out_min)) * delta + minimum
            return ((x - minimum) / delta) * (self.out_max - self.out_min) + self.out_min
        if norm == "Standardize":
            mean = float(stats["mean"])
            std = float(stats["std"])
            if std == 0.0:
                raise ValueError("cannot standardize a zero-variance channel")
            return x * std + mean if reverse else (x - mean) / std
        if norm == "LogRescale":
            minimum = float(stats["min"])
            maximum = float(stats["max"])
            delta = maximum - minimum
            if delta == 0.0:
                raise ValueError("cannot log-rescale a constant channel")
            if reverse:
                y = ((x - self.out_min) / (self.out_max - self.out_min)) * delta + minimum
                return torch.exp(y) + minimum - 1.0
            y = torch.log(x - minimum + 1.0)
            return ((y - minimum) / delta) * (self.out_max - self.out_min) + self.out_min
        raise ValueError(f"unsupported release25 normalization {norm!r}")

    def _map(self, data: Tensor, data_type: str, reverse: bool) -> Tensor:
        dim = self._channel_dim(data)
        channels = list(torch.unbind(data, dim=dim))
        for index, stats in self._stats(data_type).items():
            if index >= len(channels):
                raise ValueError(
                    f"metadata refers to channel {index}, tensor has {len(channels)} channels"
                )
            channels[index] = self._apply(channels[index], stats, reverse)
        return torch.stack(channels, dim=dim)

    def transform(self, data: Tensor, data_type: str = "Inputs") -> Tensor:
        return self._map(data, data_type, reverse=False)

    def reverse(self, data: Tensor, data_type: str = "Labels") -> Tensor:
        return self._map(data, data_type, reverse=True)


@dataclass
class Release25ModelBundle:
    role: str
    root: Path
    checkpoint: Path
    info: dict[str, Any]
    hparams: dict[str, Any]
    input_code: str
    model: torch.nn.Module
    normalizer: Release25Normalizer
    device: torch.device

    @property
    def in_channels(self) -> int:
        return len(self.info.get("Inputs", {}))

    @property
    def out_channels(self) -> int:
        return len(self.info.get("Labels", {}))


def load_standard_release25_model(
    model_dir: str | Path,
    release25_repo: str | Path,
    *,
    role: str,
    device: str | torch.device = "cpu",
) -> Release25ModelBundle:
    """Load a published standard release25 Step-1 or Step-3 model folder.

    The DaRUS archives contain the metadata needed to reconstruct ``model.pt``.
    For the standard training path release25 chooses the first value from every
    ``HPS_options.yaml`` entry; this loader follows that rule exactly. Optuna
    trial checkpoints are intentionally rejected rather than guessed.
    """

    if role not in {"cnn1", "cnn2"}:
        raise ValueError("role must be 'cnn1' or 'cnn2'")
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    info_path = _one_file(root, "info.yaml")
    hps_path = _one_file(root, "HPS_options.yaml")
    checkpoint = _checkpoint(root)
    info = _load_yaml(info_path)
    hps = _load_yaml(hps_path)

    in_channels = len(info.get("Inputs", {}))
    out_channels = len(info.get("Labels", {}))
    expected = (3, 2) if role == "cnn1" else (6, 1)
    if (in_channels, out_channels) != expected:
        raise ValueError(
            f"{role} metadata must describe {expected[0]} inputs and {expected[1]} outputs; "
            f"found {in_channels} and {out_channels}"
        )

    input_code = str(_hps_value(hps, "inputs"))
    if role == "cnn1" and len(input_code) != in_channels:
        raise ValueError(
            f"CNN1 input code {input_code!r} has {len(input_code)} symbols, expected {in_channels}"
        )

    _install_release25_import_path(release25_repo)
    networks = importlib.import_module("processing.networks.unetVariants")

    # release25 standard training uses UNet only for the special 3->1 vanilla
    # case; both published LGCNN components use UNetNoPad2.
    if in_channels == 3 and out_channels == 1:
        model = networks.UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=int(_hps_value(hps, "depth")),
            init_features=int(_hps_value(hps, "init_features")),
            kernel_size=int(_hps_value(hps, "kernel_size")),
        ).float()
    else:
        model = networks.UNetNoPad2(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=int(_hps_value(hps, "depth")),
            init_features=int(_hps_value(hps, "init_features")),
            kernel_size=int(_hps_value(hps, "kernel_size")),
            stride=int(_hps_value(hps, "stride")),
            dilation=int(_hps_value(hps, "dilation")),
            activation=str(_hps_value(hps, "activation_fct")),
            norm=str(_hps_value(hps, "norm")),
            repeat_inner=bool(_hps_value(hps, "repeat_inner")),
        ).float()

    device_obj = torch.device(device)
    try:
        state = torch.load(checkpoint, map_location=device_obj, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device_obj)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint {checkpoint} does not contain a state dict")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint does not match the architecture reconstructed from "
            f"{hps_path} and {info_path}: {exc}"
        ) from exc
    model.to(device_obj)
    model.eval()
    return Release25ModelBundle(
        role=role,
        root=root,
        checkpoint=checkpoint,
        info=info,
        hparams=hps,
        input_code=input_code,
        model=model,
        normalizer=Release25Normalizer(info),
        device=device_obj,
    )


@dataclass
class Release25FixedFields:
    material_id: Tensor
    pressure: Tensor | None = None
    pressure_gradient: Tensor | None = None
    extra_by_code: Mapping[str, Tensor] = field(default_factory=dict)

    def field_for_code(self, code: str, permeability: Tensor) -> Tensor:
        if code == "k":
            return permeability
        if code == "i":
            return self.material_id
        if code == "p":
            if self.pressure is None:
                raise ValueError("CNN1 requires pressure ('p'), but none was supplied")
            return self.pressure
        if code == "g":
            if self.pressure_gradient is None:
                raise ValueError("CNN1 requires pressure gradient ('g'), but none was supplied")
            return self.pressure_gradient
        if code in self.extra_by_code:
            return self.extra_by_code[code]
        raise ValueError(f"unsupported or missing CNN1 input code {code!r}")


@dataclass(frozen=True)
class PreparedLGCNNScenario:
    """Physical fixed fields recovered from a prepared DaRUS ``pki`` datapoint."""

    fixed: Release25FixedFields
    original_permeability: Tensor
    dataset_dir: Path
    run_id: str
    metadata: Mapping[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.original_permeability.shape)

    @classmethod
    def from_prepared_pki(
        cls,
        dataset_dir: str | Path,
        run_id: str,
        *,
        device: str | torch.device = "cpu",
    ) -> "PreparedLGCNNScenario":
        root = Path(dataset_dir).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        info = _load_yaml(root / "info.yaml")
        candidates = [root / "Inputs" / run_id, root / "Inputs" / f"{run_id}.pt"]
        inputs = [path for path in candidates if path.is_file()]
        if len(inputs) != 1:
            raise FileNotFoundError(
                f"expected exactly one prepared input for {run_id!r} below {root / 'Inputs'}"
            )
        try:
            stored = torch.load(inputs[0], map_location=device, weights_only=True)
        except TypeError:
            stored = torch.load(inputs[0], map_location=device)
        if isinstance(stored, Mapping):
            if "tensor" not in stored:
                raise TypeError(f"mapping in {inputs[0]} has no 'tensor' entry")
            stored = stored["tensor"]
        prepared = torch.as_tensor(stored, dtype=torch.float32, device=device)
        if prepared.ndim != 3:
            raise ValueError(f"prepared input must be [C,H,W], got {tuple(prepared.shape)}")
        physical = Release25Normalizer(info).reverse(prepared, "Inputs")

        def index_for(*names: str, required: bool = True) -> int | None:
            section = info.get("Inputs", {})
            for name in names:
                entry = section.get(name) if isinstance(section, Mapping) else None
                if isinstance(entry, Mapping) and "index" in entry:
                    return int(entry["index"])
            if required:
                raise KeyError(f"none of {names!r} found in prepared input metadata")
            return None

        i_idx = index_for("Material ID")
        k_idx = index_for("Permeability [m^2]", "Permeability X [m^2]")
        p_idx = index_for("Liquid Pressure [Pa]")
        g_idx = index_for("Pressure Gradient [-]", required=False)
        assert i_idx is not None and k_idx is not None and p_idx is not None

        material = physical[i_idx].round()
        unique = {int(value) for value in torch.unique(material).detach().cpu().tolist()}
        if not unique.issubset({1, 2}):
            raise ValueError(
                "prepared Material ID must use release25 convention background=1, "
                f"heat-pump=2; found {sorted(unique)}"
            )
        permeability = physical[k_idx]
        pressure = physical[p_idx]
        gradient = None if g_idx is None else physical[g_idx]
        for name, value in {
            "material_id": material,
            "permeability": permeability,
            "pressure": pressure,
            "pressure_gradient": gradient,
        }.items():
            if value is not None and value.shape != permeability.shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tuple(permeability.shape)}")
        return cls(
            fixed=Release25FixedFields(material, pressure, gradient),
            original_permeability=permeability,
            dataset_dir=root,
            run_id=run_id,
            metadata=info,
        )


def _ensure_2d(value: Tensor, name: str) -> Tensor:
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[0] == 1:
        return value[0]
    raise ValueError(f"{name} must be 2-D, got {tuple(value.shape)}")


def _unbatch(value: Tensor, channels: int, name: str) -> Tensor:
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"{name} expects batch size 1")
        value = value[0]
    if value.ndim != 3 or value.shape[0] != channels:
        raise ValueError(f"{name} must be [{channels},H,W], got {tuple(value.shape)}")
    return value


def _center_crop(value: Tensor, target: tuple[int, int]) -> Tensor:
    value = _ensure_2d(value, "field")
    h, w = value.shape
    th, tw = target
    if th > h or tw > w:
        raise ValueError(f"cannot crop {(h, w)} to {(th, tw)}")
    y0 = (h - th) // 2
    x0 = (w - tw) // 2
    return value[y0 : y0 + th, x0 : x0 + tw]


@dataclass
class Release25RuntimeAdapter:
    """Executable CNN1 -> original Step 2 -> CNN3 release25 adapter."""

    cnn1: Release25ModelBundle
    cnn2: Release25ModelBundle
    release25_repo: str | Path
    random_k: bool = True
    streamline_method: str = "RK45"

    def __post_init__(self) -> None:
        _install_release25_import_path(self.release25_repo)
        helpers = importlib.import_module("step2_streamlines.streamlines_helpers")
        self._make_streamlines = getattr(helpers, "make_streamlines")

    def _cnn1_input(self, permeability: Tensor, fixed: Release25FixedFields) -> Tensor:
        k = _ensure_2d(permeability, "permeability")
        channels: list[Tensor] = []
        for symbol in self.cnn1.input_code:
            field_value = _ensure_2d(fixed.field_for_code(symbol, k), symbol)
            if field_value.shape != k.shape:
                raise ValueError(f"CNN1 channel {symbol!r} shape does not match permeability")
            channels.append(field_value.to(self.cnn1.device, dtype=torch.float32))
        raw = torch.stack(channels, dim=0)
        return self.cnn1.normalizer.transform(raw, "Inputs").unsqueeze(0)

    def _streamlines(self, velocity: Tensor, material_id: Tensor) -> tuple[Tensor, Tensor]:
        velocity = _unbatch(velocity, 2, "physical velocity")
        cpu = velocity.detach().cpu()
        target = tuple(int(value) for value in cpu.shape[-2:])
        material = _center_crop(material_id.detach().cpu(), target)
        kwargs = {"method": self.streamline_method}
        center = self._make_streamlines(
            mat_ids=material.numpy(), vx=cpu[0].numpy(), vy=cpu[1].numpy(),
            dims=target, randomK_data=self.random_k, **kwargs,
        )
        top = self._make_streamlines(
            mat_ids=material.numpy(), vx=cpu[0].numpy(), vy=cpu[1].numpy(),
            dims=target, randomK_data=self.random_k, offset=10, **kwargs,
        )
        bottom = self._make_streamlines(
            mat_ids=material.numpy(), vx=cpu[0].numpy(), vy=cpu[1].numpy(),
            dims=target, randomK_data=self.random_k, offset=-10, **kwargs,
        )
        center = torch.as_tensor(center, dtype=velocity.dtype, device=velocity.device)
        outer = torch.as_tensor(top + bottom, dtype=velocity.dtype, device=velocity.device)
        return center, outer

    def predict(self, permeability: Tensor, fixed_inputs: Release25FixedFields) -> dict[str, Tensor]:
        k = _ensure_2d(permeability, "permeability").to(self.cnn1.device, dtype=torch.float32)
        with torch.no_grad():
            x1 = self._cnn1_input(k, fixed_inputs)
            velocity_network = _unbatch(self.cnn1.model(x1), 2, "CNN1 output")
            velocity = self.cnn1.normalizer.reverse(velocity_network, "Labels")
            center, outer = self._streamlines(velocity, fixed_inputs.material_id)

            target = tuple(int(value) for value in velocity.shape[-2:])
            k_crop = _center_crop(k, target).to(velocity.device, velocity.dtype)
            material = _center_crop(fixed_inputs.material_id, target).to(
                velocity.device, velocity.dtype
            )
            raw2 = torch.stack(
                [material, velocity[0], velocity[1], center, k_crop, outer], dim=0
            ).float()
            x2 = self.cnn2.normalizer.transform(raw2, "Inputs").unsqueeze(0)
            temperature_network = _unbatch(self.cnn2.model(x2), 1, "CNN3 output")
            temperature = self.cnn2.normalizer.reverse(temperature_network, "Labels")
        return {
            "velocity": velocity,
            "streamlines": torch.stack([center, outer], dim=0),
            "temperature": temperature,
        }


@dataclass(frozen=True)
class Release25Runtime:
    adapter: Release25RuntimeAdapter
    scenario: PreparedLGCNNScenario

    @classmethod
    def from_paths(
        cls,
        *,
        release25_repo: str | Path,
        cnn1_dir: str | Path,
        cnn2_dir: str | Path,
        prepared_pki_dir: str | Path,
        run_id: str,
        device: str | torch.device = "cpu",
        random_k: bool = True,
        streamline_method: str = "RK45",
    ) -> "Release25Runtime":
        scenario = PreparedLGCNNScenario.from_prepared_pki(
            prepared_pki_dir, run_id, device=device
        )
        cnn1 = load_standard_release25_model(
            cnn1_dir, release25_repo, role="cnn1", device=device
        )
        cnn2 = load_standard_release25_model(
            cnn2_dir, release25_repo, role="cnn2", device=device
        )
        adapter = Release25RuntimeAdapter(
            cnn1=cnn1,
            cnn2=cnn2,
            release25_repo=release25_repo,
            random_k=random_k,
            streamline_method=streamline_method,
        )
        return cls(adapter=adapter, scenario=scenario)

    def deterministic(self, permeability: Tensor | None = None) -> dict[str, Tensor]:
        k = self.scenario.original_permeability if permeability is None else permeability
        return self.adapter.predict(k, self.scenario.fixed)
