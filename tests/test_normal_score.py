import numpy as np
import yaml

from subsurface_uq.sampling import (
    ContinuousPointConditionalKLLogGaussianPermeabilityMap,
    EmpiricalNormalScoreTransform,
    KLLogGaussianPermeabilityMap,
    NormalScoreConditionalPermeabilityMap,
)


def _training_fields():
    y, x = np.mgrid[0:12, 0:14]
    fields = []
    for shift in (0.0, 0.7, 1.4):
        log10_k = (
            -9.5
            + 0.22 * np.sin((x + shift) / 3.0)
            + 0.10 * np.cos((y - shift) / 2.5)
            + 0.03 * np.sin((x + y) / 2.0)
        )
        fields.append(10.0**log10_k)
    return np.asarray(fields, dtype=np.float32)


def test_empirical_normal_score_roundtrip_preserves_training_quantiles():
    fields = _training_fields()
    transform = EmpiricalNormalScoreTransform.fit(
        fields,
        spatial_stride=1,
        n_quantiles=129,
        tail_probability=1.0e-3,
    )

    log_values = np.log10(fields)
    reconstructed = transform.from_score(transform.to_score(log_values))

    # Interior empirical support is reconstructed closely; only finite tail clipping
    # is expected at the extreme endpoints.
    assert np.sqrt(np.mean((reconstructed - log_values) ** 2)) < 0.01
    diagnostics = transform.diagnostics(fields)
    assert abs(diagnostics["score_mean"]) < 0.1
    assert 0.8 < diagnostics["score_std"] < 1.2


def test_normal_score_conditional_map_keeps_gaussian_coordinates_but_empirical_marginal():
    fields = _training_fields()
    transform = EmpiricalNormalScoreTransform.fit(
        fields,
        spatial_stride=1,
        n_quantiles=129,
        tail_probability=1.0e-3,
    )
    score_fields = transform.to_score(np.log10(fields))
    prior = KLLogGaussianPermeabilityMap(
        shape=fields.shape[1:],
        domain_size_m=(60.0, 70.0),
        mean_log10_k=float(np.mean(score_fields)),
        std_log10_k=float(np.std(score_fields, ddof=1)),
        length_scale_m=(18.0, 22.0),
        covariance_model="exponential",
        n_modes=8,
    )
    points = np.asarray([[12.0, 15.0], [42.0, 55.0]])
    observed_log10 = np.asarray([-9.55, -9.30])
    observed_scores = transform.to_score(observed_log10)
    conditional = ContinuousPointConditionalKLLogGaussianPermeabilityMap(
        prior=prior,
        observation_coordinates_yx_m=points,
        observation_log10_k=observed_scores,
        observation_std_log10_k=np.asarray([0.15, 0.15]),
    )
    wrapped = NormalScoreConditionalPermeabilityMap(transform, conditional)

    fields_out = wrapped.map_coordinates(np.zeros((2, wrapped.dimension)))
    assert fields_out.shape == (2, *fields.shape[1:])
    assert np.all(np.isfinite(fields_out))
    assert np.all(fields_out > 0.0)
    metadata = wrapped.metadata
    assert metadata["coordinate_distribution"] == "iid_standard_normal"
    assert metadata["marginal_model"] == "empirical_training_normal_score_inverse"


def test_normal_score_transform_yaml_payload_roundtrip():
    transform = EmpiricalNormalScoreTransform.fit(
        _training_fields(),
        spatial_stride=2,
        n_quantiles=65,
        tail_probability=1.0e-3,
    )
    serialized = yaml.safe_load(yaml.safe_dump(transform.to_dict()))
    reconstructed = EmpiricalNormalScoreTransform.from_dict(serialized)

    values = np.asarray([-9.7, -9.5, -9.3])
    np.testing.assert_allclose(
        reconstructed.to_score(values),
        transform.to_score(values),
    )
