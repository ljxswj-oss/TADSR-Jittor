#!/usr/bin/env python3
"""Safely write the final GitHub repository URL into submission materials.

Default behavior is dry-run/pending. The script never adds remotes and never
pushes to GitHub.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_FILES = [
    "README.md",
    "deliverables/TADSR-Jittor_submission_readme.md",
    "docs/final_github_upload_checklist.md",
    "docs/final_release_candidate_signoff.md",
    "docs/final_submission_checklist.md",
    "docs/final_human_submission_instructions.md",
    "docs/final_human_submission_lock_report.md",
]
PLACEHOLDER_LINE = "GitHub repository: <to be filled after manual upload>"
SECTION_TITLE = "## GitHub URL 状态"


def valid_url(url: str) -> bool:
    if not url:
        return False
    lowered = url.lower()
    if any(token in lowered for token in ["<user>", "<repo>", "your_", "username", "todo", "placeholder"]):
        return False
    return bool(re.match(r"^https://github\.com/[^/\s]+/[^/\s]+/?$", url) or re.match(r"^git@github\.com:[^/\s]+/[^/\s]+\.git$", url))


def upsert_url(text: str, url: str) -> str:
    new_line = f"GitHub repository: {url}"
    if "GitHub repository:" in text:
        return re.sub(r"GitHub repository:\s*.*", new_line, text)
    block = f"\n{SECTION_TITLE}\n\n{new_line}\n"
    return text.rstrip() + "\n\n" + block


def dry_run_pending(url: str | None) -> dict:
    if not url:
        return {
            "status": "PARTIAL_GITHUB_URL_PENDING",
            "reason": "No URL was provided. Create the GitHub repository manually, then rerun with --url.",
            "valid_url": False,
        }
    if not valid_url(url):
        return {
            "status": "FAIL",
            "reason": "Provided URL is empty, placeholder-like, or not a GitHub repository URL.",
            "valid_url": False,
        }
    return {"status": "PASS", "reason": "URL is valid; dry-run only, no files modified.", "valid_url": True}


def update_files(url: str) -> list[str]:
    updated: list[str] = []
    for rel in TARGET_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8", errors="ignore")
        new_text = upsert_url(original, url)
        if new_text != original:
            path.write_text(new_text, encoding="utf-8")
            updated.append(rel)
    return updated


def write_outputs(payload: dict) -> None:
    out_json = ROOT / "experiments/final_github_url_status.json"
    out_md = ROOT / "experiments/final_github_url_status.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Final GitHub URL Status",
        "",
        f"Status: **{payload['status']}**",
        "",
        "## Markers",
        "",
        "| Marker | Status |",
        "|---|---|",
    ]
    for marker, status in payload["markers"].items():
        lines.append(f"| `{marker}` | `{status}` |")
    lines += [
        "",
        f"- URL: `{payload.get('url') or '<pending>'}`",
        f"- Dry run: `{payload['dry_run']}`",
        f"- Updated files: `{len(payload['updated_files'])}`",
        "",
        "## Next Manual Step",
        "",
        payload["next_action"],
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="")
    parser.add_argument("--github-url", default="", help="Alias for --url.")
    parser.add_argument("--dry-run", default="1", choices=["0", "1"])
    args = parser.parse_args()
    url = (args.url or args.github_url).strip()
    dry_run = args.dry_run == "1"

    updated_files: list[str] = []
    check = dry_run_pending(url or None)
    status = check["status"]
    if url and check["valid_url"] and not dry_run:
        updated_files = update_files(url)
        status = "PASS"
    elif url and check["valid_url"] and dry_run:
        status = "PASS"

    next_action = (
        "Create the GitHub repository manually, then run this script with --url and --dry-run 0."
        if status == "PARTIAL_GITHUB_URL_PENDING"
        else "Review updated files and run final validators before committing."
        if status == "PASS" and not dry_run
        else "URL validation passed in dry-run mode; no repository remote was added and no push was performed."
        if status == "PASS"
        else check["reason"]
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "url": url,
        "dry_run": dry_run,
        "valid_url": bool(check["valid_url"]),
        "updated_files": updated_files,
        "reason": check["reason"],
        "next_action": next_action,
        "markers": {
            "TADSR_FINAL_GITHUB_URL_STATUS": status,
            "TADSR_FINAL_GITHUB_URL_UPDATE_SCRIPT_READY": "PASS",
        },
    }
    write_outputs(payload)
    for marker, value in payload["markers"].items():
        print(f"{marker}: {value}")
    print(f"url: {url or '<pending>'}")
    print(f"dry_run: {dry_run}")
    print(f"updated_files: {len(updated_files)}")
    return 0 if status in {"PASS", "PARTIAL_GITHUB_URL_PENDING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
