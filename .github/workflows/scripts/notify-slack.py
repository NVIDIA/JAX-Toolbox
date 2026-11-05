from __future__ import annotations

import json
import os
import sys

SLACK_CHANNEL = "#swdl-openxla-ci-health"
CC_USER_IDS = (
    "U041LNT1K7Z",
    "U7UU6GLAY",
    "U07NGSFGJ68",
    "U058YJACU7M",
)
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


def compose_text(
    *,
    tag: str,
    build_date: str,
    run_url: str,
    bullets: list[str],
) -> str:
    tag_url = f"https://github.com/NVIDIA/xla_staging/releases/tag/{tag}"
    mentions = " ".join(f"<@{uid}>" for uid in CC_USER_IDS)
    change_lines = "\n".join(f"• {b}" for b in bullets)
    return "\n".join(
        [
            f":package: *New JAX-Toolbox staging container built* for `{build_date}`",
            f"XLA staging tag: <{tag_url}|{tag}>",
            f"Workflow run: <{run_url}|GitHub Actions>",
            "",
            "*Included changes:*",
            change_lines,
            "",
            f"cc {mentions}",
        ]
    )


def write_payload_output(payload: dict) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write("PAYLOAD<<EOF_PAYLOAD\n")
        fh.write(json.dumps(payload))
        fh.write("\nEOF_PAYLOAD\n")


def main() -> int:
    tag = os.environ.get("XLA_TAG", "").strip()
    build_date = os.environ.get("BUILD_DATE", "").strip()
    tag_message = os.environ.get("XLA_TAG_MESSAGE", "")
    run_url = os.environ.get("RUN_URL", "").strip()

    if not tag:
        print("XLA_TAG is empty; cannot post Slack notification", file=sys.stderr)
        return 1

    bullets = extract_bullets(tag_message) or [NO_CHANGES_PLACEHOLDER]
    text = compose_text(
        tag=tag,
        build_date=build_date,
        run_url=run_url,
        bullets=bullets,
    )
    payload = {
        "channel": SLACK_CHANNEL,
        "text": text,
        "unfurl_links": False,
        "unfurl_media": False,
    }

    write_payload_output(payload)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
