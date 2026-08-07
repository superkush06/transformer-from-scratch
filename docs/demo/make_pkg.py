"""Bundle the package for the browser demo.

The demo ships `tfs` as a zip that Pyodide unpacks into its filesystem. That
bundle is a copy, so it can fall behind the package it was made from: change
a backward pass, forget to rebuild, and the page keeps auditing last week's
derivation while every number on it cites this week's.

    python docs/demo/make_pkg.py        # rewrites docs/demo/tfs-pkg.zip

Three assets can drift independently, so each carries its own hash:
the zip, the driver `tfsdemo.py`, and nothing else, because this page keeps
its JavaScript inline. `tests/test_demo_bundle.py` fails if any stamp goes
stale.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "tfs"
BUNDLE = ROOT / "docs" / "demo" / "tfs-pkg.zip"
STAMP = ROOT / "docs" / "demo" / "bundle.json"
DRIVER = ROOT / "docs" / "demo" / "tfsdemo.py"


def sources() -> list[pathlib.Path]:
    """Every .py in the package, in a stable order."""
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def build(out: pathlib.Path = BUNDLE) -> int:
    files = sources()
    # Deterministic: fixed timestamps and no compression jitter, so rebuilding
    # an unchanged package produces an identical file and git stays quiet.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes())
    # A content hash the page hangs off the asset URL. `cache: "no-store"` is a
    # request to revalidate, and a CDN is free to answer it from a stale copy;
    # a URL that changes when the bytes change is not.
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    driver = hashlib.sha256(DRIVER.read_bytes()).hexdigest()[:12]
    STAMP.write_text(json.dumps(
        {"sha": sha, "driver": driver, "modules": len(files)}) + "\n")
    return len(files)


if __name__ == "__main__":
    n = build()
    stamp = json.loads(STAMP.read_text())
    print(f"wrote {BUNDLE.relative_to(ROOT)} ({n} modules, "
          f"{BUNDLE.stat().st_size:,} bytes, sha {stamp['sha']}, "
          f"driver {stamp['driver']})")
