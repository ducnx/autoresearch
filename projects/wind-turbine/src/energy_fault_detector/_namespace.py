"""Helpers for exposing the existing source folders as a package."""

from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]


def extend_path(package_globals: dict, sibling_dir: str) -> None:
    """Add an existing top-level source folder to a wrapper package path."""
    target = SRC_ROOT / sibling_dir
    if target.exists():
        package_globals["__path__"].append(str(target))
