import numpy as np

from subsurface_uq.sampling import (
    TrainingCompatibilityDiagnostics,
    characterize_training_distribution,
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
