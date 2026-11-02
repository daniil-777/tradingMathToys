"""Reproduce README chart previews from a generated replay HTML (requires browser-test extra)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financeai.core.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/assets/reports"))
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    previews = []
    with sync_playwright() as api:
        browser = api.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=1)
        page.goto(args.report.resolve().as_uri())
        for theme in ("dark", "light"):
            page.select_option("#theme", theme)
            page.wait_for_function(
                "theme => window.financeaiReportReady && document.documentElement.dataset.theme === theme",
                arg=theme,
            )
            path = args.out / f"replay-{theme}.png"
            page.locator(".chart-shell").screenshot(path=str(path))
            previews.append({"file": path.name, "sha256": sha256_file(path), "theme": theme})
        data = json.loads(page.locator("#replay-data").text_content())
        manifest = {
            "schema": "financeai.replay_preview",
            "schema_version": 1,
            "html_sha256": sha256_file(args.report),
            "source_sha256": data["source_sha256"],
            "browser": browser.version,
            "viewport": {"width": 1280, "height": 1000},
            "previews": previews,
        }
        browser.close()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
