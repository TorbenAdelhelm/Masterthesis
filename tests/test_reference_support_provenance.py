import copy
from hashlib import sha256

import numpy as np
import pytest
import yaml

from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
from subsurface_uq.sampling.reference_field import (
    build_reference_field_maps, save_reference_field_input_model)


def _map():
    reference = np.power(10., -10. + np.arange(16).reshape(4, 4) / 10.)
    return build_reference_field_maps(
        reference, cell_size_m=5., residual_std_log10_k=.1,
        length_scale_m=(30., 50.), n_modes=4)[0]


def _profile(tmp_path):
    source = tmp_path / 'explicit-training.npy'
    np.save(source, np.array([[1e-12, 1e-8], [2e-12, 2e-8]]))
    return {
        'reference_level': 'release25_training_patch_primary',
        'minimum_k_m2': 1e-12, 'maximum_k_m2': 2e-8,
        'provenance': {'kind': 'explicit_training_fields', 'sources': [
            {'path': str(source), 'sha256': sha256(source.read_bytes()).hexdigest()}]},
    }


def test_nominal_reference_is_not_claimed_as_lgcnn_training_support(tmp_path):
    field_map = _map()
    path = tmp_path / 'input.yaml'
    payload = save_reference_field_input_model(path, field_map)
    assert payload['training_reference'] is None
    assert payload['reference_field']['range_m2'] == [
        float(field_map.reference_permeability.min()), float(field_map.reference_permeability.max())]
    assert load_conditional_kl_input_model(path).training_k_range is None


def test_explicit_training_profile_records_distinct_support_sources(tmp_path):
    path = tmp_path / 'input.yaml'
    profile = _profile(tmp_path)
    payload = save_reference_field_input_model(path, _map(), training_reference=profile)
    loaded = load_conditional_kl_input_model(path)
    assert loaded.training_k_range == (1e-12, 2e-8)
    assert loaded.payload['training_reference'] == profile
    assert payload['reference_field']['range_m2'] != list(loaded.training_k_range)


def test_legacy_schema3_reference_range_does_not_change_physical_law(tmp_path):
    path = tmp_path / 'input.yaml'
    field_map = _map()
    payload = save_reference_field_input_model(path, field_map)
    payload['input_law'] = 'reference-centered-lognormal-candidate'
    payload['training_reference'] = {'minimum_k_m2': 1e-11, 'maximum_k_m2': 1e-9}
    path.write_text(yaml.safe_dump(payload), encoding='utf-8')
    loaded = load_conditional_kl_input_model(path)
    assert loaded.training_k_range is None
    coordinates = np.random.default_rng(17).normal(size=(3, field_map.dimension))
    np.testing.assert_array_equal(loaded.unconditional.map_coordinates(coordinates),
                                  field_map.map_coordinates(coordinates))


@pytest.mark.parametrize('invalid', ['missing_provenance', 'empty_sources', 'bad_checksum', 'bad_range'])
def test_invalid_training_provenance_is_rejected_before_writing_and_on_load(tmp_path, invalid):
    profile = _profile(tmp_path)
    bad = copy.deepcopy(profile)
    if invalid == 'missing_provenance':
        del bad['provenance']
    elif invalid == 'empty_sources':
        bad['provenance']['sources'] = []
    elif invalid == 'bad_checksum':
        bad['provenance']['sources'][0]['sha256'] = 'invalid'
    else:
        bad['minimum_k_m2'] = -1.
    path = tmp_path / 'input.yaml'
    with pytest.raises(ValueError, match='training'):
        save_reference_field_input_model(path, _map(), training_reference=bad)
    assert not path.exists()
    if invalid == 'missing_provenance':
        return  # Provenance-free old schema-3 references intentionally remain loadable.
    payload = save_reference_field_input_model(path, _map(), training_reference=profile)
    payload['training_reference'] = bad
    path.write_text(yaml.safe_dump(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='training'):
        load_conditional_kl_input_model(path)
