from types import SimpleNamespace

import numpy as np
import pytest

from subsurface_uq.surrogates.bounded_streamlines import (
    BoundedStreamlineFactory,
    StreamlineIntegrationError,
    _bounded_calc_streamline,
    _velocity_rhs,
    configure_release25_streamlines,
)


def test_bounded_velocity_rhs_clips_rk_stage_points_to_domain():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    vx = np.array([[0.0, 1.0], [2.0, 3.0]])
    vy = np.array([[10.0, 11.0], [12.0, 13.0]])
    rhs = _velocity_rhs(x, y, vx, vy, random_k_data=True)

    # [5,-2] is projected to the boundary point [1,0]. With release25's
    # random-K axis convention the returned derivative is [vy, vx].
    value = rhs(0.0, np.array([5.0, -2.0]))

    np.testing.assert_allclose(value, np.array([12.0, 2.0]))


def test_bounded_calc_streamline_stops_at_domain_boundary():
    def rhs(_t, _state):
        return np.array([1.0, 0.0])

    sol_x, sol_y, t, nfev = _bounded_calc_streamline(
        rhs,
        (2.0, 2.0),
        np.array([1.0, 1.0]),
        t_end=10.0,
        t_steps=101,
        method="RK45",
        max_nfev=10_000,
    )

    assert len(t) < 101
    assert np.all(sol_x <= 2.0)
    assert np.all(sol_y <= 2.0)
    assert t[-1] <= 1.0
    assert nfev < 10_000


def test_bounded_calc_streamline_watchdog_fails_loudly():
    def rhs(_t, _state):
        return np.array([1.0, 0.0])

    with pytest.raises(StreamlineIntegrationError, match="max_nfev=1"):
        _bounded_calc_streamline(
            rhs,
            (100.0, 100.0),
            np.array([1.0, 1.0]),
            t_end=10.0,
            t_steps=101,
            method="RK45",
            max_nfev=1,
            diagnostic_prefix="sample=4 pass=center heat_pump=84/96",
        )


def test_configure_release25_mode_leaves_upstream_callable_untouched():
    sentinel = object()
    adapter = SimpleNamespace(_make_streamlines=sentinel)

    configure_release25_streamlines(adapter, mode="release25")

    assert adapter._make_streamlines is sentinel


def test_configure_bounded_mode_installs_factory():
    adapter = SimpleNamespace(_make_streamlines=object())

    configure_release25_streamlines(
        adapter,
        mode="bounded",
        max_nfev=1234,
        diagnostics=True,
        slow_streamline_seconds=0.5,
    )

    assert isinstance(adapter._make_streamlines, BoundedStreamlineFactory)
    assert adapter._make_streamlines.max_nfev == 1234
    assert adapter._make_streamlines.diagnostics is True
    assert adapter._make_streamlines.slow_streamline_seconds == 0.5


def test_bounded_factory_returns_finite_release25_shaped_raster():
    material = np.ones((6, 7), dtype=float)
    material[2, 3] = 2.0
    vx = np.ones_like(material)
    vy = np.zeros_like(material)
    factory = BoundedStreamlineFactory(max_nfev=10_000)

    result = factory(
        mat_ids=material,
        vx=vx,
        vy=vy,
        dims=material.shape,
        randomK_data=True,
        method="RK45",
    )

    assert tuple(result.shape) == material.shape
    assert np.all(np.isfinite(result.numpy()))
