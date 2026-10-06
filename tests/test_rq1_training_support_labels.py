from pathlib import Path
from types import SimpleNamespace

import numpy as np

from subsurface_uq.rq1.workflow import _diagnostics, _diagnostics_payload


def test_missing_artifact_training_support_has_no_perlin_fallback():
    config = SimpleNamespace(input_model=Path('input.yaml'), input_preview_count=0, conditioning_tolerance_log10=None)
    diagnostics = _diagnostics(config, conditional=False)
    diagnostics.update(np.full((2, 3, 3), 1e-10))
    result = diagnostics.finalize()
    payload = _diagnostics_payload(result, config)
    assert payload['training_range_m2'] is None
    assert payload['outside_training_fraction'] is None
    assert payload['training_range_source'] == 'unassessed_input_model_training_support'


def test_explicit_artifact_training_range_is_labelled_and_descriptive():
    config = SimpleNamespace(input_model=Path('input.yaml'), input_preview_count=0, conditioning_tolerance_log10=None)
    diagnostics = _diagnostics(config, conditional=False, training_k_range=(1e-11, 1e-9))
    diagnostics.update(np.full((2, 3, 3), 1e-10))
    payload = _diagnostics_payload(diagnostics.finalize(), config)
    assert payload['training_range_m2'] == [1e-11, 1e-9]
    assert payload['training_range_source'] == 'explicit_input_model_training_reference'
    assert payload['outside_training_fraction'] == 0


def test_manual_legacy_range_source_remains_perlin():
    config = SimpleNamespace(input_model=None, input_preview_count=0, conditioning_tolerance_log10=None)
    diagnostics = _diagnostics(config, conditional=False)
    diagnostics.update(np.full((2, 3, 3), 1e-10))
    payload = _diagnostics_payload(diagnostics.finalize(), config)
    assert payload['training_range_source'] == 'historical_release25_perlin_bounds'
    assert payload['training_range_m2'] is not None
