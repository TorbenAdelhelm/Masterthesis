from types import SimpleNamespace

from subsurface_uq.rq1.workflow import _reference_artifact_execution_plan


def _loaded(payload):
    return SimpleNamespace(payload=payload)


def test_reference_artifact_execution_plan_skips_duplicate_legacy_variants():
    unconditioned = _reference_artifact_execution_plan(_loaded({
        "schema_version": 3,
        "input_law": "reference-centered-lognormal",
        "conditioning": {},
    }))
    assert unconditioned == {
        "conditioned": False,
        "mc_variant": "B_unconditional_grf_mc",
        "rqmc_variant": "D_unconditional_grf_rqmc",
    }

    conditioned = _reference_artifact_execution_plan(_loaded({
        "schema_version": 3,
        "input_law": "reference-centered-lognormal",
        "conditioning": {"observation_coordinates_yx_m": [[1.0, 2.0]]},
    }))
    assert conditioned == {
        "conditioned": True,
        "mc_variant": "C_conditional_grf_mc",
        "rqmc_variant": "D_conditional_grf_rqmc",
    }


def test_reference_artifact_execution_plan_leaves_legacy_inputs_unchanged():
    assert _reference_artifact_execution_plan(None) is None
    assert _reference_artifact_execution_plan(_loaded({
        "schema_version": 2,
        "input_law": "legacy-lognormal",
        "conditioning": {},
    })) is None
