"""Tests for core/gee_auth.py."""

from unittest.mock import patch

import pytest

from climate_change.core.gee_auth import _resolve_project, validate_gee_project


class TestResolveProject:
    def test_explicit_project_returned_directly(self):
        result = _resolve_project("my-project", allow_prompt=False)
        assert result == "my-project"

    def test_env_var_used_when_no_explicit_project(self, monkeypatch):
        monkeypatch.setenv("GEE_PROJECT", "env-project")
        result = _resolve_project("", allow_prompt=False)
        assert result == "env-project"

    def test_missing_project_no_prompt_raises(self, monkeypatch):
        monkeypatch.delenv("GEE_PROJECT", raising=False)
        with pytest.raises(ValueError, match="GEE project ID is required"):
            _resolve_project("", allow_prompt=False)

    def test_explicit_project_takes_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("GEE_PROJECT", "env-project")
        result = _resolve_project("explicit-project", allow_prompt=False)
        assert result == "explicit-project"


class TestValidateGeeProject:
    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_gee_project("")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_gee_project("   ")

    def test_valid_project_calls_ensure_gee(self):
        with patch("climate_change.core.gee_auth.ensure_gee") as mock_ensure:
            validate_gee_project("my-project")
        mock_ensure.assert_called_once_with("my-project", allow_prompt=False)

    def test_ensure_gee_failure_raises_runtime_error(self):
        with (
            patch("climate_change.core.gee_auth.ensure_gee", side_effect=Exception("auth failed")),
            pytest.raises(RuntimeError, match="Could not authenticate"),
        ):
            validate_gee_project("bad-project")


class TestStartupInitGee:
    def test_no_gee_project_logs_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("GEE_PROJECT", raising=False)
        import logging

        from climate_change.core.gee_auth import startup_init_gee

        with caplog.at_level(logging.WARNING):
            startup_init_gee()
        assert any("GEE_PROJECT is not set" in r.message for r in caplog.records)

    def test_with_project_calls_ensure_gee(self, monkeypatch):
        monkeypatch.setenv("GEE_PROJECT", "test-project")
        with patch("climate_change.core.gee_auth.ensure_gee") as mock_ensure:
            from climate_change.core.gee_auth import startup_init_gee

            startup_init_gee()
        mock_ensure.assert_called_once_with("test-project", allow_prompt=False)

    def test_ensure_gee_failure_is_caught_and_logged(self, monkeypatch, caplog):
        monkeypatch.setenv("GEE_PROJECT", "test-project")
        import logging

        from climate_change.core.gee_auth import startup_init_gee

        with (
            patch("climate_change.core.gee_auth.ensure_gee", side_effect=Exception("fail")),
            caplog.at_level(logging.ERROR),
        ):
            startup_init_gee()
        assert any("failed" in r.message for r in caplog.records)
