from pathlib import Path


def test_geostatistical_workflow_documents_base10_std_and_no_clipping():
    text = Path("docs/geostatistical_generation_workflow.md").read_text(encoding="utf-8")
    assert "Normal(-3, 0.5^2)" in text
    assert "log10 variance is `0.25`" in text
    assert "never clipped" in text
    assert "ell_row_m" in text and "ell_col_m" in text
