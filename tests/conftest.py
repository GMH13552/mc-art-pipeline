"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from asset_tree import build_resource_root


@pytest.fixture()
def resource_root(tmp_path: Path) -> Path:
    return build_resource_root(tmp_path)
