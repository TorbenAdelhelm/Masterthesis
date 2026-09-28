import numpy as np

from subsurface_uq.sampling.new_domain import select_new_lgcnn_domain


def test_auto_new_domain_maximizes_measurement_count_and_aligns_cells():
    coordinates = np.asarray(
        [
            [100.0, 100.0],
            [200.0, 200.0],
            [300.0, 300.0],
            [900.0, 900.0],
            [2500.0, 2500.0],
        ],
        dtype=np.float64,
    )
    domain, inside = select_new_lgcnn_domain(
        coordinates, domain_size_m=1000.0, cell_size_m=100.0
    )

    assert domain.shape == (10, 10)
    assert domain.conditioning_measurement_count == 4
    assert int(np.count_nonzero(inside)) == 4
    assert np.isclose(domain.west_edge_m % 100.0, 0.0)
    assert np.isclose(domain.south_edge_m % 100.0, 0.0)


def test_explicit_new_domain_uses_projected_origin_and_local_coordinates():
    coordinates = np.asarray(
        [[1000.0, 2000.0], [1200.0, 2200.0], [3000.0, 4000.0]],
        dtype=np.float64,
    )
    domain, inside = select_new_lgcnn_domain(
        coordinates,
        domain_size_m=1000.0,
        cell_size_m=100.0,
        west_edge_m=900.0,
        south_edge_m=1900.0,
    )

    np.testing.assert_array_equal(inside, [True, True, False])
    local = domain.projected_xy_to_local_yx(
        coordinates[inside, 0], coordinates[inside, 1]
    )
    np.testing.assert_allclose(local, [[100.0, 100.0], [300.0, 300.0]])
    assert domain.first_cell_center_x_m == 950.0
    assert domain.first_cell_center_y_m == 1950.0
    assert domain.east_edge_m == 1900.0
    assert domain.north_edge_m == 2900.0


def test_explicit_new_domain_requires_both_origin_coordinates():
    coordinates = np.asarray([[0.0, 0.0], [100.0, 100.0]])
    try:
        select_new_lgcnn_domain(
            coordinates,
            domain_size_m=1000.0,
            cell_size_m=100.0,
            west_edge_m=0.0,
        )
    except ValueError as exc:
        assert "both west_edge_m and south_edge_m" in str(exc)
    else:
        raise AssertionError("partial explicit origin should fail")
