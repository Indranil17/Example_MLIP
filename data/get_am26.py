#!/usr/bin/env python
"""Download the AM26 extxyz file and record or verify its checksum.

Usual use:   python data/get_am26.py          (downloads, then verifies against the committed CHECKSUMS.json)
Re-record:   python data/get_am26.py --record (rewrites CHECKSUMS.json; only when the source file is meant to change)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RAW_URL = "https://raw.githubusercontent.com/vldgroup/AM26/main/data/am26.extxyz"
COMMIT_API = "https://api.github.com/repos/vldgroup/AM26/commits/main"
HERE = Path(__file__).resolve().parent
CHECKSUMS = HERE / "CHECKSUMS.json"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def latest_commit() -> str | None:
    try:
        with urllib.request.urlopen(COMMIT_API, timeout=20) as r:  # noqa: S310
            return json.load(r).get("sha")
    except Exception as exc:  # network or rate limit; not fatal
        print(f"[get_am26] could not query GitHub API for the commit sha: {exc}")
        return None


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get_am26] downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(target, "wb") as fh:  # noqa: S310
        total = 0
        for block in iter(lambda: r.read(1 << 20), b""):
            fh.write(block)
            total += len(block)
    print(f"[get_am26] wrote {target} ({total / 1e6:.1f} MB)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(HERE / "am26.extxyz"))
    p.add_argument("--url", default=RAW_URL)
    p.add_argument("--record", action="store_true", help="write CHECKSUMS.json instead of verifying against it")
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    a = p.parse_args()

    target = Path(a.out)
    if a.force or not target.exists():
        download(a.url, target)
    digest = sha256_of(target)
    print(f"[get_am26] sha256 {digest}")

    if a.record:
        rec = {
            "file": target.name,
            "url": a.url,
            "sha256": digest,
            "bytes": target.stat().st_size,
            "commit_main_at_download": latest_commit(),
            "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        CHECKSUMS.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"[get_am26] recorded {CHECKSUMS}")
        return 0

    if not CHECKSUMS.exists():
        print("[get_am26] no CHECKSUMS.json; run once with --record", file=sys.stderr)
        return 2
    rec = json.loads(CHECKSUMS.read_text(encoding="utf-8"))
    if rec["sha256"] != digest:
        print(f"[get_am26] CHECKSUM MISMATCH: recorded {rec['sha256']}, got {digest}", file=sys.stderr)
        return 1
    print("[get_am26] checksum verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
