"""Tests for core/http.py — get_with_retry."""

from unittest.mock import patch

import pytest
import requests

from climate_change.core.http import get_with_retry


def test_returns_response_on_first_success():
    fake_resp = object()
    with patch("climate_change.core.http.requests.get", return_value=fake_resp) as mock_get:
        result = get_with_retry("https://example.com/band.tif", timeout=10)
    assert result is fake_resp
    mock_get.assert_called_once_with("https://example.com/band.tif", timeout=10)


def test_retries_on_connection_error_then_succeeds():
    fake_resp = object()
    with (
        patch(
            "climate_change.core.http.requests.get",
            side_effect=[requests.exceptions.ConnectionError("dropped"), fake_resp],
        ) as mock_get,
        patch("climate_change.core.http.time.sleep") as mock_sleep,
    ):
        result = get_with_retry("https://example.com/band.tif", timeout=10, attempts=3, backoff=0.1)
    assert result is fake_resp
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once()


def test_retries_on_timeout_then_succeeds():
    fake_resp = object()
    with (
        patch(
            "climate_change.core.http.requests.get",
            side_effect=[requests.exceptions.Timeout("slow"), fake_resp],
        ) as mock_get,
        patch("climate_change.core.http.time.sleep"),
    ):
        result = get_with_retry("https://example.com/band.tif", timeout=10, attempts=3, backoff=0.1)
    assert result is fake_resp
    assert mock_get.call_count == 2


def test_raises_after_exhausting_all_attempts():
    with (
        patch(
            "climate_change.core.http.requests.get",
            side_effect=requests.exceptions.ConnectionError("still down"),
        ) as mock_get,
        patch("climate_change.core.http.time.sleep"),
        pytest.raises(requests.exceptions.ConnectionError),
    ):
        get_with_retry("https://example.com/band.tif", timeout=10, attempts=3, backoff=0.1)
    assert mock_get.call_count == 3


def test_does_not_retry_non_transient_exceptions():
    """A non-2xx status doesn't raise from requests.get itself — callers call
    resp.raise_for_status() separately — but any other exception type (not
    ConnectionError/Timeout) should propagate immediately, unretried."""
    with (
        patch(
            "climate_change.core.http.requests.get",
            side_effect=ValueError("not a network issue"),
        ) as mock_get,
        pytest.raises(ValueError),
    ):
        get_with_retry("https://example.com/band.tif", timeout=10, attempts=3, backoff=0.1)
    assert mock_get.call_count == 1
