import numpy as np
import yaml

from subsurface_uq.sampling import load_conditional_kl_input_model


def test_stochastic_input_model_roundtrip_reconstructs_gaussian_coordinate_map(tmp_path):
    path = tmp_path / "stochastic_input_model.yaml"
    payload = {
        "schema_version": 1,
        "coordinate_distribution": "iid_standard_normal",
        "coordinate_dimension": 5,
        "field_shape": [3, 4],
        "prior": {
            "map": "KLLogGaussianPermeabilityMap",
            "log_space": "log10",
            "covariance": "separable_exponential",
            "shape": [3, 4],
            "domain_size_m": [300.0, 400.0],
            "mean_log10_k": -9.4,
            "std_log10_k": 0.25,
            "global_mean_std_log10_k": 0.08,
            "length_scale_m": [90.0, 140.0],
            "dimension": 5,
            "spatial_dimension": 4,
            "selection": "fixed_n_modes",
            "requested_n_modes": 4,
            "energy_threshold_requested": 0.95,
            "retained_energy_fraction": 0.5,
        },
        "training_reference": {
            "minimum_k_m2": 1.0e-11,
            "maximum_k_m2": 5.0e-9,
        },
        "conditioning": {
            "observation_coordinates_yx_m": [[50.0, 50.0], [250.0, 350.0]],
            "observation_log10_intrinsic_permeability": [-9.6, -9.1],
            "observation_std_log10_k": [0.15, 0.15],
        },
        "source_policy": {
            "prior": "actual_lgcnn_training_permeability_fields",
            "conditioning": "real_munich_measurements",
            "sample_filtering": None,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = load_conditional_kl_input_model(path)

    assert loaded.prior.field_shape == (3, 4)
    assert loaded.prior.spatial_dimension == 4
    assert loaded.conditional.dimension == 5
    assert loaded.training_k_range == (1.0e-11, 5.0e-9)
    fields = loaded.conditional.map_coordinates(np.zeros((2, 5)))
    assert fields.shape == (2, 3, 4)
    assert np.all(np.isfinite(fields))
    assert np.all(fields > 0.0)
    assert (
        loaded.conditional.metadata["coordinate_distribution"]
        == "iid_standard_normal"
    )
