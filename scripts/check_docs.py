"""Check repository-relative Markdown file links without network access."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!\w)!?\[[^\]\n]+\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)")


def main() -> int:
    failures = []
    checked = 0
    for document in sorted([*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")]):
        fenced = False
        for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if fenced:
                continue
            for match in LINK.finditer(line):
                url = urlsplit(match.group(1).strip("<>"))
                if url.scheme or url.netloc or not url.path:
                    continue
                checked += 1
                target = (document.parent / unquote(url.path)).resolve()
                if not target.is_relative_to(ROOT) or not target.exists():
                    failures.append(f"{document.relative_to(ROOT)}:{number}: unavailable link {match.group(1)}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Checked {checked} repository-relative Markdown links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
