"""Tests for core/dask_engine.py."""

from unittest.mock import MagicMock

import pytest

from climate_change.core.dask_engine import DaskEngine


class TestRunIoParallel:
    def test_runs_all_tasks_and_returns_results(self):
        tasks = {"a": lambda: 1, "b": lambda: 2}
        result = DaskEngine.run_io_parallel(tasks)
        assert result == {"a": 1, "b": 2}

    def test_empty_tasks_returns_empty_dict(self):
        assert DaskEngine.run_io_parallel({}) == {}

    def test_exception_in_task_propagates(self):
        def _boom():
            raise ValueError("task error")

        with pytest.raises(ValueError, match="task error"):
            DaskEngine.run_io_parallel({"boom": _boom})

    def test_max_workers_param_accepted(self):
        tasks = {"x": lambda: 99}
        result = DaskEngine.run_io_parallel(tasks, max_workers=1)
        assert result == {"x": 99}

    def test_multiple_tasks_all_executed(self):
        results_seen = []
        tasks = {str(i): (lambda i=i: results_seen.append(i) or i) for i in range(5)}
        out = DaskEngine.run_io_parallel(tasks)
        assert len(out) == 5
        assert sorted(out.values()) == list(range(5))


class TestComputeWithProgress:
    def test_calls_compute_on_lazy_result(self):
        mock_lazy = MagicMock()
        mock_lazy.compute.return_value = "materialised"
        result = DaskEngine.compute_with_progress(mock_lazy)
        mock_lazy.compute.assert_called_once()
        assert result == "materialised"


class TestGetClientIfRunning:
    def test_returns_none_when_no_cluster_started(self):
        DaskEngine._client = None
        assert DaskEngine.get_client_if_running() is None

    def test_returns_none_when_client_closed(self):
        mock_client = MagicMock()
        mock_client.status = "closed"
        DaskEngine._client = mock_client
        assert DaskEngine.get_client_if_running() is None
        assert DaskEngine._client is None

    def test_returns_active_client(self):
        mock_client = MagicMock()
        mock_client.status = "running"
        DaskEngine._client = mock_client
        result = DaskEngine.get_client_if_running()
        assert result is mock_client
        DaskEngine._client = None  # cleanup


class TestShutdown:
    def test_shutdown_closes_client(self):
        mock_client = MagicMock()
        DaskEngine._client = mock_client
        DaskEngine.shutdown()
        mock_client.close.assert_called_once()
        assert DaskEngine._client is None

    def test_shutdown_when_no_client_is_safe(self):
        DaskEngine._client = None
        DaskEngine.shutdown()  # must not raise


class TestClipRasterToAoi:
    def test_clips_with_polygon(self, simple_polygon_geojson):
        mock_da = MagicMock()
        mock_da.rio.clip.return_value = "clipped"
        result = DaskEngine.clip_raster_to_aoi(mock_da, simple_polygon_geojson)
        assert result == "clipped"

    def test_clips_with_feature(self, feature_geojson):
        mock_da = MagicMock()
        mock_da.rio.clip.return_value = "clipped_feat"
        result = DaskEngine.clip_raster_to_aoi(mock_da, feature_geojson)
        assert result == "clipped_feat"
