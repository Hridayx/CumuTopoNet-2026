#!/usr/bin/env python3
"""Check that the repository contains source files only."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP = {
    "README.md", "THIRD_PARTY.md", ".gitignore", "pyproject.toml",
    "requirements-core.lock", "configs", "docs", "scripts", "src", "tests",
}
FORBIDDEN_DIRS = {
    ".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist", "data",
    "work", "results", "outputs", "checkpoints", "logs",
}
FORBIDDEN_SUFFIXES = {
    ".pyc", ".pyo", ".dat", ".zip", ".pt", ".pth", ".ckpt", ".npy",
    ".npz", ".pdf", ".png", ".log", ".pem", ".key",
}


def audit():
    problems = []
    files = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in ALLOWED_TOP:
            problems.append(f"Unexpected top-level item: {rel}")
        if any(part in FORBIDDEN_DIRS or part.endswith(".egg-info") for part in rel.parts):
            problems.append(f"Generated or private directory: {rel}")
        if path.is_symlink():
            problems.append(f"Symlink is not portable: {rel}")
        if not path.is_file():
            continue
        files.append(path)
        if path.suffix in FORBIDDEN_SUFFIXES or path.name.startswith(".env"):
            problems.append(f"Generated or private file: {rel}")
        if path.stat().st_size > 5_000_000:
            problems.append(f"Unexpected file larger than 5 MB: {rel}")
        if rel.parts[0] == "configs" and path.suffix == ".yaml" and not path.name.endswith(".example.yaml"):
            problems.append(f"Local configuration: {rel}")
    return sorted(set(problems)), files


def main():
    problems, files = audit()
    if problems:
        print("\n".join(problems))
        return 1
    print(f"Repository check passed: {len(files)} source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
