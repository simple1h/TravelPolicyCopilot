from __future__ import annotations

import pytest

from src.config import settings


@pytest.fixture(autouse=True)
def default_test_backend_mode(monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "test")
