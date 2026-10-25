"""Verify and unpack the explicit, immutable inputs to the v5 paper analysis."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(root: Path) -> dict:
    return json.loads((root / "provenance/analysis_inputs.json").read_text())


def _destination(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid artifact member path: {relative!r}")
    destination = root.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact member escapes the release: {relative!r}")
    return destination


def _check_file(path: Path, expected: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen analysis input: {path}. Run scripts/reproduce_paper.py first.")
    if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"Frozen analysis input differs from the release manifest: {path}")


def extract_results(root: Path) -> None:
    """Extract only manifest-listed regular files, refusing conflicting files."""
    manifest = _manifest(root)
    archive_path = _destination(root, manifest["archive"]["path"])
    _check_file(archive_path, manifest["archive"])
    expected = {member["path"]: member for member in manifest["members"]}
    if len(expected) != len(manifest["members"]):
        raise ValueError("Duplicate paths in the frozen input manifest")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(expected):
            raise ValueError("Archive member inventory differs from the frozen manifest")
        for member in members:
            if not member.isfile():
                raise ValueError(f"Expected a regular file in results archive: {member.name}")
            destination = _destination(root, member.name)
            if destination.exists():
                _check_file(destination, expected[member.name])
                continue
            with archive.extractfile(member) as handle:
                payload = handle.read()
            specification = expected[member.name]
            if len(payload) != specification["size"] or hashlib.sha256(payload).hexdigest() != specification["sha256"]:
                raise ValueError(f"Result archive payload failed verification: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(payload)


def frozen_input_paths(root: Path, selection: str) -> list[Path]:
    """Return frozen ordering; new result directories cannot affect the paper."""
    paths = [_destination(root, relative) for relative in _manifest(root)[selection]]
    if not paths:
        raise ValueError(f"Empty frozen input selection: {selection}")
    return paths


def verify_inputs(root: Path) -> None:
    for specification in _manifest(root)["members"]:
        _check_file(_destination(root, specification["path"]), specification)
    metadata = json.loads((root / "provenance/analysis_metadata.json").read_text())
    for specification in metadata["files"]:
        _check_file(_destination(root, specification["path"]), specification)
