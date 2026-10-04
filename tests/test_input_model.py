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



def test_schema2_normal_score_input_model_reconstructs_physical_map(tmp_path):
    path = tmp_path / "stochastic_input_model_v2.yaml"
    probabilities = [0.001, 0.25, 0.5, 0.75, 0.999]
    log10_quantiles = [-10.2, -9.7, -9.4, -9.1, -8.7]
    payload = {
        "schema_version": 2,
        "input_law": "normal-score-copula",
        "coordinate_distribution": "iid_standard_normal",
        "coordinate_dimension": 4,
        "field_shape": [3, 4],
        "prior": {
            "map": "KLLogGaussianPermeabilityMap",
            "log_space": "log10",
            "covariance": "separable_exponential",
            "shape": [3, 4],
            "domain_size_m": [300.0, 400.0],
            "mean_log10_k": 0.0,
            "std_log10_k": 1.0,
            "global_mean_std_log10_k": 0.0,
            "length_scale_m": [90.0, 140.0],
            "dimension": 4,
            "spatial_dimension": 4,
            "selection": "fixed_n_modes",
            "requested_n_modes": 4,
            "energy_threshold_requested": 0.95,
            "retained_energy_fraction": 0.5,
        },
        "normal_score_transform": {
            "transform": "empirical_normal_score",
            "physical_variable": "log10_intrinsic_permeability_m2",
            "latent_variable": "standard_normal_score",
            "probabilities": probabilities,
            "log10_quantiles": log10_quantiles,
            "fit_spatial_stride": 1,
            "source_value_count": 100,
            "requested_quantile_count": 5,
            "tail_probability": 0.001,
            "tail_policy": "clip_to_empirical_quantile_support",
        },
        "training_reference": {
            "minimum_k_m2": 1.0e-11,
            "maximum_k_m2": 5.0e-9,
        },
        "conditioning": {
            "space": "empirical_normal_score",
            "observation_coordinates_yx_m": [[50.0, 50.0], [250.0, 350.0]],
            "observation_log10_intrinsic_permeability": [-9.6, -9.1],
            "observation_latent_values": [-0.4, 0.7],
            "observation_std_log10_k": [0.15, 0.15],
            "observation_std_latent": [0.2, 0.2],
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = load_conditional_kl_input_model(path)

    assert loaded.prior.field_shape == (3, 4)
    assert loaded.conditional.dimension == 4
    fields = loaded.conditional.map_coordinates(np.zeros((2, 4)))
    assert fields.shape == (2, 3, 4)
    assert np.all(np.isfinite(fields))
    assert np.all(fields > 0.0)
    assert loaded.conditional.metadata["marginal_model"] == (
        "empirical_training_normal_score_inverse"
    )



def test_schema2_measurement_kriging_input_model_reconstructs_gaussian_map(tmp_path):
    path = tmp_path / "measurement_kriging_input_model.yaml"
    payload = {
        "schema_version": 2,
        "input_law": "measurement-kriging",
        "coordinate_distribution": "iid_standard_normal",
        "coordinate_dimension": 4,
        "field_shape": [3, 4],
        "prior": {
            "map": "KLLogGaussianPermeabilityMap",
            "log_space": "log10",
            "covariance": "separable_exponential",
            "shape": [3, 4],
            "domain_size_m": [300.0, 400.0],
            "mean_log10_k": -9.35,
            "std_log10_k": 0.28,
            "global_mean_std_log10_k": 0.0,
            "length_scale_m": [90.0, 140.0],
            "dimension": 4,
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
            "space": "log10_intrinsic_permeability",
            "observation_coordinates_yx_m": [[50.0, 50.0], [250.0, 350.0]],
            "observation_log10_intrinsic_permeability": [-9.6, -9.1],
            "observation_latent_values": [-9.6, -9.1],
            "observation_std_log10_k": [0.15, 0.15],
            "observation_std_latent": [0.15, 0.15],
        },
        "source_policy": {
            "prior": "real_munich_measurement_variogram_spatial_cv",
            "conditioning": "real_munich_measurements_in_prior_space",
            "training_fields_role": "lgcnn_support_and_fidelity_reference",
            "sample_filtering": None,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = load_conditional_kl_input_model(path)

    assert loaded.prior.mean_log10_k == -9.35
    assert loaded.conditional.dimension == 4
    fields = loaded.conditional.map_coordinates(np.zeros((2, 4)))
    assert fields.shape == (2, 3, 4)
    assert np.all(np.isfinite(fields))
    assert np.all(fields > 0.0)
    assert loaded.payload["source_policy"]["prior"] == (
        "real_munich_measurement_variogram_spatial_cv"
    )
