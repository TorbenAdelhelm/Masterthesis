import numpy as np

from subsurface_uq.sampling.spatial_diagnostics import log_field_spectrum, compare_reference_ensemble


def test_spectral_power_satisfies_parseval_and_detects_directional_texture():
    yy, xx = np.indices((32, 32))
    log_field = -9.5 + .3 * np.sin(2*np.pi*xx*6/32)
    field = 10**log_field
    spectrum = log_field_spectrum(field, cell_size_m=20)
    taper = np.outer(np.hanning(32), np.hanning(32))
    expected = np.mean(((log_field - log_field.mean()) * taper)**2) / np.mean(taper**2)
    np.testing.assert_allclose(np.sum(spectrum["power_2d"]), expected, rtol=1e-12)
    assert spectrum["axis_band_power_fraction"] > .8
    rotated = log_field_spectrum(field.T, cell_size_m=20)
    np.testing.assert_allclose(rotated["radial_power"], spectrum["radial_power"], atol=1e-14)
    assert np.argmax(rotated["angular_power"]) != np.argmax(spectrum["angular_power"])


def test_identity_ensemble_has_zero_matched_support_error():
    yy, xx = np.indices((16, 16))
    reference = 10**(-9.5 + .03*xx + .05*yy + .1*np.sin(xx))
    result = compare_reference_ensemble(reference, np.stack([reference, reference]), cell_size_m=5)
    assert result["marginal"]["wasserstein_distance_log10"] == 0
    assert result["spectra"]["radial_power_relative_l2"] == 0
    assert result["spectra"]["angular_power_relative_l2"] == 0
    assert set(result["both_diagonal_variograms"]) == {"positive", "negative"}
