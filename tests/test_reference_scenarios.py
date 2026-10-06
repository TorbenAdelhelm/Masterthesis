import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from subsurface_uq.sampling.scenarios import (
    ReferenceScenario, physical_to_lognormal, lognormal_to_physical,
    bounds_to_lognormal, lognormal_to_bounds, scenario_matrix, scenario_weights)
from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
from subsurface_uq.sampling.reference_field import save_reference_field_input_model
from subsurface_uq.experiments.reference_scenarios import run_matrix
from subsurface_uq.experiments.reference_kl_sensitivity import compare_qoi_truncations


def test_physical_and_quantile_roundtrips_and_coverage():
    mean, variance = 2e-10, 3e-21
    mu, sigma = physical_to_lognormal(mean, variance)
    np.testing.assert_allclose(lognormal_to_physical(mu, sigma), [mean, variance], rtol=1e-12, atol=0)
    lb, ub = lognormal_to_bounds(mu, sigma, .01)
    np.testing.assert_allclose(bounds_to_lognormal(lb, ub, .01), [mu, sigma], rtol=1e-12)
    samples = np.exp(mu + sigma * np.random.default_rng(12).normal(size=100000))
    assert abs(np.mean((samples >= lb) & (samples <= ub)) - .99) < .001
    assert np.any(samples < lb) and np.any(samples > ub)
    np.testing.assert_allclose(physical_to_lognormal(2., 0), [np.log(2), 0])


@pytest.mark.parametrize("call,args", [
    (physical_to_lognormal, (0., 1.)), (physical_to_lognormal, (1., -1.)),
    (lognormal_to_physical, (0., -1.)), (bounds_to_lognormal, (1., 2., 0.)),
    (bounds_to_lognormal, (2., 1., .01)), (lognormal_to_bounds, (0., 1., 1.)),
    (physical_to_lognormal, (float('nan'), 1.))])
def test_invalid_marginals(call, args):
    with pytest.raises(ValueError):
        call(*args)


def test_expert_specs_reject_conflicts_and_derive_log10_units():
    mu, sigma = np.log(2e-10), .3
    lb, ub = lognormal_to_bounds(mu, sigma, .01)
    scenario = ReferenceScenario('RUN_1', lb=lb, ub=ub, alpha=.01, marginal_reference_k=2e-10)
    assert scenario.sigma_R == pytest.approx(sigma / np.log(10))
    assert scenario.manifest()['anchor_log_parameters']['mu_ln'] == pytest.approx(mu)
    mean, var = lognormal_to_physical(mu, sigma)
    ReferenceScenario('RUN_1', center='arithmetic-mean', physical_mean=mean,
                      physical_variance=var, marginal_reference_k=mean)
    for overrides in ({'sigma_R': 1.}, {'marginal_reference_k': 1e-10},
                      {'physical_mean': mean, 'physical_variance': var}, {'ub': None}):
        params = dict(reference_run='RUN_1', lb=lb, ub=ub, alpha=.01, marginal_reference_k=2e-10)
        params.update(overrides)
        with pytest.raises(ValueError):
            ReferenceScenario(**params)
    with pytest.raises(ValueError, match='exactly one'):
        ReferenceScenario('RUN_1', sigma_R=.1, n_modes=5)


def test_matrix_ids_are_order_independent_and_weights_are_explicit():
    base = ReferenceScenario('RUN_1', sigma_R=.1)
    axes = {'reference_run': ['RUN_1', 'RUN_2'], 'sigma_R': [.05, .1],
            'truncation': [{'n_modes': 2}, {'energy_threshold': .99}]}
    rows = scenario_matrix(base, axes)
    assert len(rows) == 8
    assert {r.scenario_id for r in rows} == {r.scenario_id for r in scenario_matrix(base, dict(reversed(list(axes.items()))))}
    ids = [r.scenario_id for r in rows]
    assert scenario_weights(ids) is None
    assert sum(scenario_weights(ids, 'equal').values()) == 1
    with pytest.raises(ValueError):
        scenario_weights(ids, {ids[0]: 1.})
    with pytest.raises(ValueError):
        scenario_matrix(base, {'sigma_R': [.1, .1]})


def test_matrix_artifacts_patch_diagnostics_and_legacy_loading(tmp_path):
    field = 10**(-9 + np.arange(64).reshape(8, 8) / 100)
    np.save(tmp_path / 'run.npy', field)
    config = {'base': {'reference_run': 'RUN_1', 'sigma_R': .05},
              'axes': {'truncation': [{'n_modes': 3}, {'energy_threshold': .99}]},
              'references': {'RUN_1': 'run.npy'}, 'cell_size_m': 5.,
              'permeability_convention': 'historical-training', 'n_samples': 4,
              'training_support': {'reference_npy': ['run.npy'], 'box_size': 4,
                                   'skip_per_dir': 2, 'feature_stride': 1}}
    source = tmp_path / 'matrix.yaml'
    source.write_text(yaml.safe_dump(config))
    report = run_matrix(source, tmp_path / 'out')
    assert report['pooling'] is False and report['scenario_weights'] is None
    assert len(report['scenarios']) == 2
    for row in report['scenarios']:
        path = tmp_path / 'out' / row['scenario_id'] / 'stochastic_input_model.yaml'
        loaded = load_conditional_kl_input_model(path)
        assert loaded.unconditional.metadata['scenario']['scenario_id'] == row['scenario_id']
        diagnostics = json.loads(path.with_name('diagnostics.json').read_text())
        assert diagnostics['training_patch_support']['generated_field_count'] == 4
        assert diagnostics['compatibility_assessment']['sample_filtering'] is None
        payload = yaml.safe_load(path.read_text())
        payload['input_law'] = 'reference-centered-lognormal-candidate'
        path.write_text(yaml.safe_dump(payload))
        assert load_conditional_kl_input_model(path).prior.dimension == loaded.prior.dimension
        payload['prior']['std_log10_k'] = .2
        path.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError, match='conflicts'):
            load_conditional_kl_input_model(path)


@pytest.mark.parametrize('method', ['MC', 'RQMC'])
def test_paired_temperature_qoi_sensitivity_is_reproducible(method):
    reference = np.full((4, 5), 1e-10)
    maps = [ReferenceScenario('RUN_1', sigma_R=.1, n_modes=n, energy_threshold=None).build_maps(
        reference, cell_size_m=5.)[0] for n in [2, 8, 20]]
    def qoi(field):
        # Deterministic test temperature response with a local and global QoI.
        temperature = 10.6 + np.log10(field / 1e-10)
        return [temperature[0, 0], temperature.mean() - 10.6]
    report, values = compare_qoi_truncations(maps, qoi, n_samples=32, method=method)
    repeat, repeated = compare_qoi_truncations(maps, qoi, n_samples=32, method=method)
    np.testing.assert_array_equal(values, repeated)
    assert report == repeat
    assert report['truncations'][-1]['paired_rmse_to_largest'] == [0., 0.]
    assert report['truncations'][0]['paired_rmse_to_largest'][0] > 0
    with pytest.raises(ValueError, match='differing only'):
        other = ReferenceScenario('RUN_2', sigma_R=.2).build_maps(reference, cell_size_m=5.)[0]
        compare_qoi_truncations([maps[0], other], qoi, n_samples=4)


def test_temperature_kl_cli_queries_adapter_and_records_qoi_context(tmp_path, monkeypatch):
    import torch
    from subsurface_uq.experiments.reference_kl_sensitivity import main
    from subsurface_uq.rq1 import config as config_module
    from subsurface_uq.surrogates.release25_runtime import Release25Runtime
    from subsurface_uq.surrogates import bounded_streamlines
    reference = np.full((4, 4), 1e-10)
    paths = []
    for n in [2, 8]:
        field_map = ReferenceScenario('RUN_1', sigma_R=.1, n_modes=n, energy_threshold=None).build_maps(
            reference, cell_size_m=5.)[0]
        path = tmp_path / f'input_{n}.yaml'
        save_reference_field_input_model(path, field_map)
        paths.append(str(path))
    calls = []
    class Adapter:
        def predict(self, permeability, fixed):
            calls.append(permeability.shape)
            return {'temperature': 20.6 + torch.log10(permeability)}
    config = SimpleNamespace(release25_repo='repo', cnn1_dir='cnn1', cnn2_dir='cnn2', prepared_pki_dir='pki',
                             fixed_run_id='RUN_1', device='cpu', random_k=False, streamline_method='RK45',
                             streamline_mode='release25', streamline_max_nfev=10000, streamline_diagnostics=False,
                             streamline_slow_seconds=2., cell_size_m=5., background_temperature=10.6,
                             receptors=((0, 0),), mean_anomaly_roi=(0, 4, 0, 4), to_dict=lambda: {'test': True})
    monkeypatch.setattr(config_module, 'load_rq1_config', lambda path: config)
    monkeypatch.setattr(Release25Runtime, 'from_paths', lambda **kwargs: SimpleNamespace(
        adapter=Adapter(), scenario=SimpleNamespace(shape=(4, 4), fixed=None)))
    monkeypatch.setattr(bounded_streamlines, 'configure_release25_streamlines', lambda *args, **kwargs: None)
    assert main(['--input-models', *paths, '--rq1-config', 'config.yaml', '--n-samples', '4',
                 '--output-dir', str(tmp_path / 'qoi')]) == 0
    assert len(calls) == 8
    assert np.load(tmp_path / 'qoi' / 'paired_temperature_qoi.npy').shape == (2, 4, 2)
    report = json.loads((tmp_path / 'qoi' / 'qoi_truncation.json').read_text())
    assert report['qoi_names'] == ['temperature_0_0', 'mean_temperature_anomaly']


def test_single_reference_cli_explicit_mode_counts(tmp_path):
    from subsurface_uq.experiments.reference_field_permeability import main
    np.save(tmp_path / 'ref.npy', np.full((4, 4), 1e-10))
    assert main(['--reference-npy', str(tmp_path / 'ref.npy'), '--reference-run', 'RUN_1',
                 '--permeability-convention', 'historical-training', '--residual-std-log10-k', '.05',
                 '--length-scale-y-m', '50', '--length-scale-x-m', '80', '--n-modes', '2', '4',
                 '--n-samples', '4', '--output-dir', str(tmp_path / 'out')]) == 0
    loaded = load_conditional_kl_input_model(tmp_path / 'out' / 'modes_4' / 'stochastic_input_model.yaml')
    assert loaded.unconditional.dimension == 4
    assert loaded.payload['scenario']['config']['energy_threshold'] is None
