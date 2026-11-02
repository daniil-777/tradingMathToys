"""Real Chromium checks; enable explicitly with FINANCEAI_BROWSER_TESTS=1."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from financeai.reporting import ReplayReportConfig, write_replay_report

from .test_replay_report import mixed as mixed

pytestmark = pytest.mark.skipif(os.environ.get("FINANCEAI_BROWSER_TESTS") != "1", reason="optional browser suite")


@pytest.fixture
def page():
    api = pytest.importorskip("playwright.sync_api")
    with api.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        yield page
        browser.close()


def ready(page, instrument=None):
    page.wait_for_function(
        "window.financeaiReportReady === true || document.getElementById('render-error').textContent"
    )
    assert page.locator("#render-error").inner_text() == ""
    if instrument:
        page.wait_for_function(
            "key => document.getElementById('replay-chart').dataset.instrument === key", arg=instrument
        )


def test_offline_controls_download_hover_and_mobile_layout(page, mixed, tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    result, prep, scenario = mixed
    artifact = write_replay_report(result, prep, tmp_path / "report.html", scenario=scenario)
    errors = []
    requests = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "request", lambda request: requests.append(request.url) if request.url.startswith(("http:", "https:")) else None
    )
    page.goto(artifact.html_path.as_uri())
    ready(page)
    assert not errors and not requests
    assert page.locator("#account-status").inner_text().startswith("VALID")
    assert page.locator("#currency").input_value() == "CHF"
    assert page.locator("#replay-chart").evaluate("el=>el.layout.shapes[1].y0") == 0
    assert "shorting disabled" in page.locator("#instrument-info").inner_text()
    assert page.locator("#replay-chart").evaluate("el=>el.data[8].visible") == "legendonly"
    page.evaluate("Plotly.restyle('replay-chart', {visible:true}, [8])")
    inverse = prep.data.instruments[2].key
    page.select_option("#instrument", inverse)
    ready(page, inverse)
    assert page.locator("#currency").input_value() == "COIN"
    assert page.locator("#replay-chart").evaluate("el=>el.layout.shapes[1].y0") < 0
    assert page.locator("#replay-chart").evaluate("el=>el.data[8].visible") is True
    page.select_option("#theme", "light")
    page.wait_for_function("document.documentElement.dataset.theme === 'light'")
    ready(page)
    # Zoom is synchronized by matching x axes; a selector change preserves the chosen interval.
    page.evaluate("Plotly.relayout('replay-chart', {'xaxis.range':[0.003,0.020]})")
    page.select_option("#currency", "CHF")
    page.wait_for_function("document.getElementById('replay-chart').dataset.currency === 'CHF'")
    assert page.locator("#replay-chart").evaluate("el=>el.layout.xaxis.range") == [0.003, 0.020]
    page.click("#reset-view")
    page.wait_for_function("document.getElementById('replay-chart').layout.xaxis.autorange === true")
    # Hover retains the exact epoch-nanosecond string instead of a lossy JS integer.
    page.evaluate("Plotly.Fx.hover('replay-chart',[{curveNumber:2,pointNumber:0}])")
    assert "epoch ns" in page.locator(".hoverlayer").text_content()
    assert "Order delay" in page.locator(".hoverlayer").text_content()
    with page.expect_download() as download:
        page.click("#download-data")
    downloaded = Path(download.value.path())
    data = json.loads(downloaded.read_text())
    assert isinstance(data["start_ns"], str)
    assert len(data["accounts"]) == 2
    # Native selectors remain keyboard-accessible, and narrow screens do not overflow the page.
    page.locator("#theme").focus()
    assert page.locator("#theme").evaluate("el=>el===document.activeElement")
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_function("document.documentElement.scrollWidth <= window.innerWidth")
    assert not errors and not requests


def test_untrusted_titles_and_invalid_accounts_are_visible_without_script_execution(
    page, mixed, tmp_path: Path
) -> None:
    pytest.importorskip("plotly")
    result, prep, scenario = mixed
    # A stricter display audit makes the full account invalid; no currency gets a terminal score.
    from financeai.domains.market import RiskLimits

    title = "Review </script><script>window.BAD=1</script>"
    artifact = write_replay_report(
        result,
        prep,
        tmp_path / "invalid.html",
        scenario=scenario,
        limits=RiskLimits(1),
        config=ReplayReportConfig(title=title),
    )
    page.goto(artifact.html_path.as_uri())
    ready(page)
    assert page.locator("h1").inner_text() == title
    assert page.evaluate("window.BAD === undefined")
    assert page.locator("#account-status").inner_text().startswith("INVALID")
    assert page.locator("#net").inner_text() == "Unavailable"
