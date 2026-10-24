# Tests for weather.py. No real network calls -- requests.get is mocked,
# since get_weather must never crash or hang the API if Open-Meteo is down.
import os
import sys
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import weather as weather_module  # noqa: E402
from weather import get_weather  # noqa: E402


def setup_function(_):
    weather_module._cache.clear()  # tests must not see each other's cached results


def test_get_weather_returns_none_on_timeout():
    with patch("weather.requests.get", side_effect=requests.exceptions.Timeout):
        assert get_weather(20.0, 74.0) is None


def test_get_weather_returns_none_on_bad_json():
    bad_response = MagicMock()
    bad_response.raise_for_status.return_value = None
    bad_response.json.side_effect = ValueError("not JSON")
    with patch("weather.requests.get", return_value=bad_response):
        assert get_weather(20.1, 74.1) is None


def test_get_weather_returns_none_when_lat_or_lon_missing():
    assert get_weather(None, None) is None
