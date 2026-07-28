"""Tests for core/gee_auth.py."""

from unittest.mock import patch

import pytest

from climate_change.core.gee_auth import _resolve_project, ensure_gee, validate_gee_project


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


class TestEnsureGeeAuthPath:
    """ensure_gee must prefer direct ee.Initialize() over
    drought_monitoring.gee.authenticate() (interactive OAuth) whenever a
    service-account key is configured — see gee_auth.py's comment for why:
    that wrapper unconditionally calls the interactive ee.Authenticate()
    first, which fails outright in a headless server with no browser/TTY,
    even when a valid service-account key is already mounted."""

    def test_service_account_present_calls_ee_initialize_directly(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/secrets/gee-service-account.json")
        with (
            patch("ee.Initialize") as mock_ee_init,
            patch("drought_monitoring.gee.authenticate") as mock_dm_auth,
        ):
            ensure_gee("service-account-project", allow_prompt=False)
        mock_ee_init.assert_called_once_with(project="service-account-project")
        mock_dm_auth.assert_not_called()

    def test_no_service_account_uses_drought_monitoring_authenticate(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        with (
            patch("ee.Initialize") as mock_ee_init,
            patch("drought_monitoring.gee.authenticate") as mock_dm_auth,
        ):
            ensure_gee("interactive-project", allow_prompt=False)
        mock_dm_auth.assert_called_once_with(project="interactive-project", quiet=True)
        mock_ee_init.assert_not_called()

    def test_service_account_initialise_failure_raises_clear_runtime_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/secrets/gee-service-account.json")
        with (
            patch("ee.Initialize", side_effect=Exception("bad key")),
            pytest.raises(RuntimeError, match="service-account credentials"),
        ):
            ensure_gee("broken-service-account-project", allow_prompt=False)


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
