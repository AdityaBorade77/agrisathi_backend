# Shared test fixtures for backend/tests. Applies to every test file here.
import pytest

from models import sarvam_client


@pytest.fixture(autouse=True)
def _no_real_sarvam_calls(tmp_path, monkeypatch):
    """Keep every test from touching the real Sarvam API or the real
    backend/data/sarvam_usage.json -- even though backend/.env may have a
    real SARVAM_API_KEY set (for manual scripts like scripts/check_sarvam.py
    and scripts/compare_pipelines.py). Without this, any test that reaches
    /api/crop for mr/hi would try the real Sarvam path first and burn real
    daily-cap requests just from running the test suite.

    Runs before any test-specific fixture that re-enables it (e.g.
    test_crop_provider.py's `sarvam_configured`), since pytest runs autouse
    fixtures before explicitly-requested fixtures of the same scope.
    """
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    monkeypatch.setattr(sarvam_client, "USAGE_PATH", str(tmp_path / "sarvam_usage.json"))
