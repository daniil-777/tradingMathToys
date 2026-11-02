"""Portable HTML research dashboards with escaped content and no external assets."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_STYLE = """
:root {color-scheme:dark; --bg:#0b1220; --panel:#111e31; --line:#26364a; --muted:#a9b9cb;}
* {box-sizing:border-box} body {margin:0; background:var(--bg); color:#f3f5f8;
font:16px/1.6 system-ui,sans-serif} main {max-width:1120px; margin:auto; padding:48px 24px}
h1 {font-size:clamp(28px,5vw,46px); line-height:1.2} .eyebrow {color:#53dcb4; letter-spacing:.12em}
.cards {display:flex; gap:16px; flex-wrap:wrap; margin:32px 0}
.card,section {background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:22px}
.card {flex:1; min-width:180px} .card small,footer {color:var(--muted)} .card strong {display:block; font-size:24px}
section {margin:20px 0} h2 {margin-top:0} details {padding:8px 0} summary {cursor:pointer; color:#80b7ff}
.scroll {overflow:auto} table {width:100%; border-collapse:collapse; font-size:14px}
th,td {text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top}
th {color:#a9b9cb} pre {white-space:pre-wrap; overflow-wrap:anywhere} dl {display:grid;
grid-template-columns:minmax(120px,1fr) 3fr; gap:10px} dd {margin:0; overflow-wrap:anywhere}
footer {margin-top:32px} :focus-visible {outline:2px solid #f1cf83; outline-offset:4px}
"""


def _text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
    return html.escape(text)


def _section(value: Any, depth: int = 0) -> str:
    if isinstance(value, list) and value and all(isinstance(row, Mapping) for row in value):
        keys = list(dict.fromkeys(key for row in value for key in row))
        header = "".join(f"<th scope='col'>{_text(key)}</th>" for key in keys)
        rows = "".join("<tr>" + "".join(f"<td>{_text(row.get(key))}</td>" for key in keys) + "</tr>" for row in value)
        return f"<div class='scroll'><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"
    if isinstance(value, Mapping) and depth < 5:
        parts = []
        for key, item in value.items():
            if isinstance(item, Mapping | list):
                parts.append(f"<details><summary>{_text(key)}</summary>{_section(item, depth + 1)}</details>")
            else:
                parts.append(f"<dl><dt>{_text(key)}</dt><dd>{_text(item)}</dd></dl>")
        return "".join(parts)
    return f"<pre>{html.escape(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))}</pre>"


def render_dashboard_html(payload: Mapping[str, Any]) -> str:
    """Render ``title``, scalar ``summary`` cards, and arbitrary named ``sections``.

    Sections may contain records, mappings, or JSON values. Nested sections are collapsible.
    Units and evidence scope belong in the supplied headings and records; none are inferred.
    """
    if not isinstance(payload, Mapping) or not isinstance(payload.get("sections"), Mapping) or not payload["sections"]:
        raise ValueError("a dashboard requires nonempty named sections")
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    title = _text(payload.get("title", "Research dashboard"))
    cards = "".join(
        f"<div class='card'><small>{_text(key)}</small><strong>{_text(value)}</strong></div>"
        for key, value in summary.items()
    )
    sections = "".join(
        f"<section><h2>{_text(key)}</h2>{_section(value)}</section>" for key, value in payload["sections"].items()
    )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{_STYLE}</style></head><body><main>"
        f"<div class='eyebrow'>FINANCEAI / RESEARCH</div><h1>{title}</h1>"
        f"<div class='cards'>{cards}</div>{sections}"
        "<footer>Generated from the supplied records. Interpret results within their stated model and evidence scope."
        "</footer></main></body></html>"
    )


def write_dashboard(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a self-contained dashboard after validating the complete payload."""
    content = render_dashboard_html(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


__all__ = ["render_dashboard_html", "write_dashboard"]
