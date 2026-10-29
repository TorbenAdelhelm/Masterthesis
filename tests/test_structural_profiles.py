from dataclasses import replace
import numpy as np
import pytest

from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec,GSToolsReferenceFFTSampler
from subsurface_uq.sampling.structural_profiles import (grid_mean_correlation,rectangle_quadrature,
    approximate_grid_mean_correlation,canonical_spec,variogram_curves,structural_features,fit_profile)


def test_exact_grid_mean_correlation_matches_independent_dense_matrix_and_variance():
    shape=(7,8);spec=GSToolsReferenceSpec('matern32_radial',13.,23.,sigma_R=.3,angle_deg=31.)
    points=np.column_stack(np.unravel_index(np.arange(np.prod(shape)),shape))*5.
    offsets=points[:,None]-points[None,:]
    r=spec.correlation(offsets[...,0],offsets[...,1])
    q=grid_mean_correlation(spec,shape,5.,chunk_rows=3)
    assert q==pytest.approx(r.mean(),abs=1e-15)
    # E population variance after removing each realization's own sample mean.
    centering=np.eye(len(r))-np.ones_like(r)/len(r)
    assert spec.sigma_R**2*(1-q)==pytest.approx(np.trace(centering@(.3**2*r))/len(r),abs=1e-15)


def test_grid_average_correlation_rotated_canonicalization_and_large_grid_quadrature():
    spec=GSToolsReferenceSpec('matern32_radial',2000.,1100.,sigma_R=.2,angle_deg=-40.)
    canonical=canonical_spec(spec)
    assert canonical.ell_col_m>=canonical.ell_row_m
    rng=np.random.default_rng(3);a=rng.normal(size=(100,2))*3000
    np.testing.assert_allclose(spec.correlation(*a.T),canonical.correlation(*a.T),atol=1e-15)
    shape=(200,180);cell=5.
    q=grid_mean_correlation(spec,shape,cell)
    approx=approximate_grid_mean_correlation(spec,rectangle_quadrature(shape,cell,48))
    assert approx==pytest.approx(q,abs=1e-5)


def test_directional_variograms_include_native_and_both_diagonals():
    a=np.arange(12*13,dtype=float).reshape(12,13)/100
    curves=variogram_curves(a,[1,2,4])
    assert curves['row'][0]==pytest.approx(.5*.13**2)
    assert curves['column'][0]==pytest.approx(.5*.01**2)
    assert curves['diagonal_positive'][0]==pytest.approx(.5*.14**2)
    assert curves['diagonal_negative'][0]==pytest.approx(.5*.12**2)


def test_profile_fits_total_variance_and_does_not_fix_realization_moments():
    shape=(35,36);spec=GSToolsReferenceSpec('matern32_radial',25.,45.,sigma_R=.2,angle_deg=20.)
    sampler=GSToolsReferenceFFTSampler(spec,shape,cell_size_m=5.,workers=1)
    logs=-9.3+sampler.sample_residual(10)
    profile=fit_profile(logs,cell_size_m=5.,lags_cells=[1,2,4,8],length_bounds_m=[5.,300.],quadrature_order=48)
    law=profile['law'];q=profile['grid_mean_unit_variance'];sigma=law['covariance']['sigma_R']
    assert law['mean_log10_k']==pytest.approx(logs.mean())
    assert sigma**2*(1-q)==pytest.approx(logs.var(),abs=1e-14)
    assert sigma>logs.std()
    assert not profile['length_at_bound']


def test_connectivity_and_fixed_thresholds_do_not_force_a_histogram():
    a=np.full((20,20),-9.3);a[:,8:11]=-8.
    f=structural_features(a,cell_size_m=5.,mean_log10=-9.3,sigma_log10=.2,stride=1)
    assert f['high_spans_rows']
    assert not f['high_spans_columns']
    assert f['high_fraction']==pytest.approx(.15)
    assert f['high_largest_component_fraction']==pytest.approx(.15)
    assert f['low_fraction']==0.


def test_explicit_fft_embedding_is_recorded_and_defines_shared_linear_driving_noise():
    shape=(5,6);a=GSToolsReferenceSpec('matern32_radial',8.,10.,sigma_R=1.)
    b=replace(a,sigma_R=2.)
    first=GSToolsReferenceFFTSampler(a,shape,cell_size_m=5.,workers=1,embedding_shape=[32,32])
    second=GSToolsReferenceFFTSampler(b,shape,cell_size_m=5.,workers=1,embedding_shape=[32,32])
    assert first.metadata['embedding_shape']==[32,32]
    np.testing.assert_array_equal(second.sample_residual(4),2*first.sample_residual(4))
    with pytest.raises(ValueError,match='powers of two'):
        GSToolsReferenceFFTSampler(a,shape,cell_size_m=5.,workers=1,embedding_shape=[8,16])


def test_profile_artifact_paths_are_portable_and_driving_law_is_unambiguous(tmp_path):
    import json
    import yaml
    from subsurface_uq.experiments.structural_profiles import fit,validate,verify_artifact
    from subsurface_uq.io import sha256_file
    maps=tmp_path/'maps';maps.mkdir();configuration=tmp_path/'configuration';configuration.mkdir()
    spec=GSToolsReferenceSpec('matern32_radial',5.,8.,sigma_R=.1)
    sampler=GSToolsReferenceFFTSampler(spec,(9,9),cell_size_m=5.,workers=1)
    path=maps/'source.npy';np.save(path,10**(-9.3+sampler.sample_residual(34)))
    initial_hash=sha256_file(path)
    cfg={'references':{'example':'../maps/source.npy'},'preparation_manifest':None,'cell_size_m':5.,
        'permeability_units':'m2','lags_cells':[1,2,4],'length_bounds_m':[2.,12.],
        'quadrature_order':16,'structural_samples':4,'diagnostic_stride_cells':1,
        'paired_embedding_shape':[32,32],'seed':31,'workers':1}
    source=configuration/'fit.yaml';source.write_text(yaml.safe_dump(cfg))
    artifact=tmp_path/'profiles/catalog.json';report=tmp_path/'report'
    p=fit(source,artifact,report)
    assert '\\' not in p['source_maps']['example']['path']
    assert all('\\' not in key for key in p['source_code_sha256'])
    # A Windows-authored catalog must remain readable on POSIX hosts too.
    p['source_maps']['example']['path']=p['source_maps']['example']['path'].replace('/','\\')
    artifact.write_text(json.dumps(p));verify_artifact(artifact)
    result=validate(artifact,report)
    metadata=result['profiles']['example_structural']['sampler_metadata']
    assert metadata['spec']['marginal']=='zero-mean unit Gaussian driving field'
    assert 'reference' not in metadata['spec']['marginal']
    assert sha256_file(path)==initial_hash
