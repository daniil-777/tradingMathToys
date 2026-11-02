"""Dashboard records remain portable and cannot inject active HTML."""

from pathlib import Path

import pytest

from financeai.research.dashboard import render_dashboard_html, write_dashboard


def test_dynamic_market_records_and_write(tmp_path: Path) -> None:
    payload = {
        "title": "Synthetic CHF research",
        "summary": {"Currency": "CHF", "Runs": 2},
        "sections": {
            "Replays": [{"name": "trend", "net_pnl": 1.25}, {"name": "flow", "net_pnl": -0.5}],
            "Assumptions": {"instrument": {"type": "spot", "shortable": False}},
        },
    }
    content = render_dashboard_html(payload)
    assert "<!DOCTYPE html>" in content and "Synthetic CHF research" in content
    assert "trend" in content and "<details>" in content and "<table>" in content
    target = write_dashboard(payload, tmp_path / "reports" / "index.html")
    assert target.read_text() == content


def test_dashboard_escapes_untrusted_record_content() -> None:
    attack = "</title><script>alert(1)</script>"
    content = render_dashboard_html({"title": attack, "sections": {attack: [{"name": attack}]}})
    assert "<script>" not in content
    assert "&lt;script&gt;" in content


@pytest.mark.parametrize("payload", [{}, {"sections": {}}, {"sections": []}, {"sections": {"x": {}}, "summary": []}])
def test_dashboard_rejects_malformed_payloads(payload) -> None:
    with pytest.raises(ValueError):
        render_dashboard_html(payload)
