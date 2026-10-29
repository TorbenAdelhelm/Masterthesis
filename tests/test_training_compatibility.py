import numpy as np

from subsurface_uq.sampling import (
    TrainingCompatibilityDiagnostics,
    RELEASE25_REALK_TRAINING_DATA_DOI,
    TrainingPatchCompatibilityDiagnostics,
    characterize_training_distribution,
    characterize_training_patch_distribution,
    permeability_ensemble_fidelity,
    release25_patch_positions,
)


def _fields():
    y, x = np.mgrid[0:8, 0:10]
    return np.stack(
        [
            10.0
            ** (
                -9.5
                + 0.10 * np.sin((x + shift) / 2.5)
                + 0.06 * np.cos(y / 2.0)
            )
            for shift in (0.0, 0.7, 1.4)
        ]
    ).astype(np.float32)


def test_training_distribution_profile_uses_actual_fields_and_spatial_descriptors():
    fields = _fields()
    profile = characterize_training_distribution(
        fields,
        cell_size_m=5.0,
        spatial_stride=1,
    )

    assert profile.field_count == 3
    assert profile.field_shape == (8, 10)
    assert profile.minimum_k == float(np.min(fields))
    assert profile.maximum_k == float(np.max(fields))
    assert "gradient_rms_x_log10_per_m" in profile.metric_reference
    assert "lag1_correlation_y" in profile.metric_reference


def test_training_compatibility_is_diagnostic_only_and_keeps_every_generated_field():
    fields = _fields()
    profile = characterize_training_distribution(fields, cell_size_m=5.0)
    diagnostics = TrainingCompatibilityDiagnostics(profile)
    generated = np.stack(
        [
            fields[0],
            fields[1] * np.float32(1.02),
            fields[2] * np.float32(1.30),
        ]
    )

    diagnostics.update(generated)
    result = diagnostics.finalize()

    assert diagnostics.count == 3
    assert result["generated_field_count"] == 3
    assert result["decision_rule"] is None
    assert (
        result["metrics"]["mean_log10_k"][
            "fraction_generated_fields_outside_training_envelope"
        ]
        >= 0.0
    )
    assert "not used to reject" in result["training_reference"]["interpretation"]



def test_release25_patch_positions_match_simulation_dataset_cuts_indexing():
    positions = release25_patch_positions(
        (12, 12),
        box_size=4,
        skip_per_dir=2,
    )

    # release25 uses (H-B)*(W-B)//skip^2 = 16 patches per full field.
    assert positions.shape == (16, 2)
    np.testing.assert_array_equal(
        positions[:5],
        np.asarray([[0, 0], [0, 2], [0, 4], [0, 6], [2, 0]]),
    )
    np.testing.assert_array_equal(positions[-1], [6, 6])


def test_patch_profile_treats_overlapping_cutouts_as_correlated_support_not_new_fields():
    fields = _fields()
    profile = characterize_training_patch_distribution(
        fields,
        cell_size_m=5.0,
        box_size=4,
        skip_per_dir=2,
        max_patches=9,
        feature_stride=1,
        field_names=("RUN_1", "RUN_2", "RUN_3"),
    )

    assert profile.field_count == 3
    assert profile.patch_population_per_field == 6
    assert profile.patch_population_total == 18
    assert profile.sampled_patch_count == 9
    assert profile.field_names == ("RUN_1", "RUN_2", "RUN_3")
    payload = profile.to_dict()
    assert payload["reference_level"] == "release25_training_patch_primary"
    assert payload["training_data_doi"] == RELEASE25_REALK_TRAINING_DATA_DOI
    assert payload["training_data_doi"] == "10.18419/DARUS-5065"
    assert "not treated as independent geological" in payload["interpretation"]


def test_patch_compatibility_is_primary_diagnostic_and_never_filters_samples():
    fields = _fields()
    profile = characterize_training_patch_distribution(
        fields,
        cell_size_m=5.0,
        box_size=4,
        skip_per_dir=2,
        max_patches=9,
        feature_stride=1,
    )
    diagnostics = TrainingPatchCompatibilityDiagnostics(
        profile,
        generated_patches_per_field=2,
    )

    generated = np.stack([fields[0], fields[1] * np.float32(1.15)])
    diagnostics.update(generated)
    result = diagnostics.finalize()

    assert result["reference_level"] == "release25_training_patch_primary"
    assert result["generated_field_count"] == 2
    assert result["generated_patch_count"] == 4
    assert result["decision_rule"] is None
    assert "No generated field or patch is rejected" in result["interpretation"]



def test_ensemble_fidelity_reports_marginal_and_variogram_mismatch_without_filtering():
    training = _fields()
    generated = training * np.float32(1.05)

    result = permeability_ensemble_fidelity(
        training,
        generated,
        cell_size_m=5.0,
        spatial_stride=1,
        max_lag_cells=3,
    )

    assert result["marginal"]["wasserstein_distance_log10"] > 0.0
    assert set(result["directional_variograms"]) == {"x", "y", "diag"}
    assert result["decision_rule"] is None
