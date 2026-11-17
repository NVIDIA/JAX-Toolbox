from __future__ import annotations

import json
import os
import sys

ANNOUNCEMENT_FILE = "announcement.json"
NO_CHANGES_PLACEHOLDER = "(no stacked MR merge commits since origin/main)"


def extract_bullets(tag_message: str) -> list[str]:
    bullets: list[str] = []
    in_section = False
    for raw in tag_message.splitlines():
        stripped = raw.rstrip().strip()
        if not in_section:
            if stripped.lower().startswith("included changes"):
                in_section = True
            continue
        if not stripped:
            if bullets:
                break
            continue
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
        elif stripped.startswith("-"):
            bullets.append(stripped[1:].strip())
        else:
            break
    return bullets


def compose_announcement(*, tag: str, build_date: str, bullets: list[str]) -> dict:
    tag_url = f"https://github.com/NVIDIA/xla_staging/releases/tag/{tag}"
    return {
        "title": f"New JAX-Toolbox staging container built for {build_date}",
        "sections": [
            {"body": f"XLA staging tag: [{tag}]({tag_url})"},
            {
                "heading": "Included changes",
                "body": "\n".join(f"• {b}" for b in bullets),
            },
        ],
    }


def main() -> int:
    tag = os.environ.get("XLA_TAG", "").strip()
    build_date = os.environ.get("BUILD_DATE", "").strip()
    tag_message = os.environ.get("XLA_TAG_MESSAGE", "")

    if not tag:
        print("XLA_TAG is empty; cannot compose the announcement", file=sys.stderr)
        return 1

    bullets = extract_bullets(tag_message) or [NO_CHANGES_PLACEHOLDER]
    announcement = compose_announcement(tag=tag, build_date=build_date, bullets=bullets)
    with open(ANNOUNCEMENT_FILE, "w", encoding="utf-8") as fh:
        json.dump(announcement, fh, indent=2)
    print(json.dumps(announcement, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
