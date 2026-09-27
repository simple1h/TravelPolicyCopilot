from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings


@pytest.mark.parametrize("mode", ["strict", "demo_fallback", "test"])
def test_backend_mode_accepts_only_known_values(mode: str):
    settings = Settings(backend_mode=mode)
    assert settings.backend_mode == mode


@pytest.mark.parametrize("mode", ["", "strcit", "unknown"])
def test_backend_mode_rejects_invalid_values(mode: str):
    with pytest.raises(ValidationError):
        Settings(backend_mode=mode)
