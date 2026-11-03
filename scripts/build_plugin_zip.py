#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and validate the QGIS plugin archive for 02H3Forge.

Usage::

    python3 scripts/build_plugin_zip.py            # -> dist/zero2h3forge-<version>.zip

The archive contains a single top-level ``zero2h3forge/`` folder, honours
``.zipignore`` (one path prefix or glob per line) and never ships VCS data,
caches, build output or this script. Basic plugins.qgis.org requirements are
checked before the zip is written; any problem aborts with exit code 1.
"""
from __future__ import annotations

import configparser
import fnmatch
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "zero2h3forge"
ALWAYS_EXCLUDE = [
    ".git", ".github", ".idea", ".vscode", "__pycache__", "*.pyc", "*.pyo",
    ".pytest_cache", ".ruff_cache", "dist", "scripts", ".gitignore", ".zipignore",
    "*.swp", ".DS_Store", "Thumbs.db",
]
REQUIRED_METADATA = [
    "name", "qgisMinimumVersion", "description", "about", "version",
    "author", "email", "repository",
]


def load_ignore_patterns() -> list:
    patterns = list(ALWAYS_EXCLUDE)
    zipignore = ROOT / ".zipignore"
    if zipignore.exists():
        for line in zipignore.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line.rstrip("/"))
    return patterns


def is_excluded(rel: Path, patterns: list) -> bool:
    parts = rel.parts
    for pat in patterns:
        for i in range(len(parts)):
            if fnmatch.fnmatch(parts[i], pat) or fnmatch.fnmatch("/".join(parts[: i + 1]), pat):
                return True
    return False


def read_metadata() -> configparser.SectionProxy:
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = str  # keep camelCase keys
    cp.read(ROOT / "metadata.txt", encoding="utf-8")
    return cp["general"]


def validate(files: list) -> list:
    problems = []
    meta_text = (ROOT / "metadata.txt").read_text(encoding="utf-8")
    if "%" in meta_text:
        problems.append("metadata.txt contains a literal '%' (breaks QGIS ConfigParser).")
    meta = read_metadata()
    for key in REQUIRED_METADATA:
        if not meta.get(key, "").strip():
            problems.append(f"metadata.txt is missing required key '{key}'.")
    init_src = (ROOT / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'PLUGIN_VERSION\s*=\s*"([^"]+)"', init_src)
    if not m or m.group(1) != meta.get("version"):
        problems.append(
            f"__init__.PLUGIN_VERSION ({m.group(1) if m else 'missing'}) does not match "
            f"metadata version ({meta.get('version')})."
        )
    icon = meta.get("icon", "")
    if icon and not (ROOT / icon).exists():
        problems.append(f"Icon '{icon}' referenced in metadata.txt does not exist.")
    if meta.get("hasProcessingProvider", "no") == "yes" and not (ROOT / "h3forge_processing.py").exists():
        problems.append("hasProcessingProvider=yes but the provider module is missing.")
    rels = {f.as_posix() for f in files}
    for required in ("__init__.py", "metadata.txt", "LICENSE"):
        if required not in rels:
            problems.append(f"Archive would not contain '{required}'.")
    for f in files:
        if f.suffix in (".pyc", ".pyo") or "__pycache__" in f.parts:
            problems.append(f"Compiled Python file would be shipped: {f}")
    return problems


def main() -> int:
    patterns = load_ignore_patterns()
    files = sorted(
        p.relative_to(ROOT)
        for p in ROOT.rglob("*")
        if p.is_file() and not is_excluded(p.relative_to(ROOT), patterns)
    )
    problems = validate(files)
    if problems:
        print("Plugin validation failed:")
        for p in problems:
            print(f"  - {p}")
        return 1

    version = read_metadata()["version"]
    out_dir = ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{PACKAGE}-{version}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(ROOT / rel, f"{PACKAGE}/{rel.as_posix()}")
    print(f"Built {out.relative_to(ROOT)} ({len(files)} files, {out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
