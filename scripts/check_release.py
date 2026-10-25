#!/usr/bin/env python3
"""Verify the package inventory and frozen audio hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/release_files.json"
EXCLUDED = {".git", ".scratch", ".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def inventory() -> dict:
    records = {}
    model_dirs = {path.parent.name for path in ROOT.glob("configs/*/config.json")}
    synthetic_audio = {record["path"]: record for record in
                       json.loads((ROOT / "provenance/dataset_audio.json").read_text())["files"]
                       if record["group"] == "synthetic"}

    def excluded(relative: Path) -> bool:
        if EXCLUDED.intersection(relative.parts):
            return True
        return (relative.parts[0] == "runs" or relative.parts[:2] == ("analysis", "outputs")
                or len(relative.parts) >= 2 and relative.parts[0] in model_dirs
                and relative.parts[1].startswith("results"))

    paths = []
    for directory, directories, files in os.walk(ROOT, followlinks=False):
        directories[:] = sorted(name for name in directories
                                if not excluded((Path(directory) / name).relative_to(ROOT)))
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"Release contains a symlink: {Path(directory) / name}")
        paths.extend(Path(directory) / name for name in files
                     if not excluded((Path(directory) / name).relative_to(ROOT)))
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            raise ValueError(f"Release contains a symlink: {relative}")
        if not path.is_file() or path == MANIFEST:
            continue
        if ((path.suffix.lower() in {".wav", ".flac", ".mp3"}
                and relative.as_posix() not in synthetic_audio)
                or path.suffix.lower() == ".pyc"
                or path.name == ".DS_Store" or path.suffix.lower() == ".env"
                or path.name.startswith(".env") and path.name != ".env.example"
                or ".env" in relative.parts or "__MACOSX" in relative.parts):
            raise ValueError(f"Unexpected private/generated artifact: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        records[relative.as_posix()] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        if relative.as_posix() in synthetic_audio:
            expected = synthetic_audio[relative.as_posix()]
            if records[relative.as_posix()] != {key: expected[key] for key in ("sha256", "bytes")}:
                raise ValueError(f"Frozen synthetic audio mismatch: {relative}")
    missing_audio = sorted(synthetic_audio.keys() - records.keys())
    if missing_audio:
        raise ValueError(f"Missing frozen synthetic audio: {missing_audio}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    actual = inventory()
    if args.write_manifest:
        MANIFEST.write_text(json.dumps({"schema_version": 1, "files": actual}, indent=2) + "\n")
    else:
        expected = json.loads(MANIFEST.read_text())["files"]
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        changed = sorted(key for key in actual.keys() & expected.keys() if actual[key] != expected[key])
        if missing or extra or changed:
            raise ValueError(f"Inventory mismatch: missing={missing}, extra={extra}, changed={changed}")
    print(f"Verified {len(actual)} package files and SHA-256 digests.")


if __name__ == "__main__":
    main()
