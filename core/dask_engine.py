from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, cast

from dask.distributed import Client, Future, LocalCluster


class DaskEngine:
    """
    Manages a Dask LocalCluster for distributed in-memory computation.
    Singleton pattern — one cluster per process.

    Two tiers of parallelism:
    - Dask Futures (submit / gather)  : CPU-bound tasks (ML training, xarray ops).
    - I/O thread pool (run_io_parallel): GEE HTTP downloads that share a session.
    """

    _client: Client | None = None

    @classmethod
    def get_client(cls) -> Client:
        """Return the singleton Dask Client, starting a LocalCluster if needed."""
        if cls._client is None or cls._client.status == "closed":
            cluster = LocalCluster(
                n_workers=int(os.environ.get("DASK_WORKERS", 4)),
                threads_per_worker=int(os.environ.get("DASK_THREADS_PER_WORKER", 2)),
                memory_limit=os.environ.get("DASK_MEMORY_LIMIT", "4GB"),
                silence_logs=True,
            )
            cls._client = Client(cluster)
        return cls._client

    @classmethod
    def get_client_if_running(cls) -> Client | None:
        """Return the active Dask Client, or None if the cluster has not been started."""
        if cls._client is not None and cls._client.status == "closed":
            cls._client = None
        return cls._client

    @classmethod
    def shutdown(cls) -> None:
        """Shut down the cluster and release resources."""
        if cls._client:
            cls._client.close()
            cls._client = None

    # ── Dask distributed helpers (CPU-bound tasks) ───────────────────────────

    @classmethod
    def submit(cls, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        """Submit a callable to the Dask cluster. Returns a Future."""
        return cls.get_client().submit(fn, *args, pure=False, **kwargs)

    @classmethod
    def gather(cls, futures: list) -> list:
        """Block until all futures complete and return results in submission order."""
        return cast(list, cls.get_client().gather(futures))

    # ── Thread-pool I/O helpers (GEE network calls) ──────────────────────────

    @staticmethod
    def run_io_parallel(
        tasks: dict[str, Callable[[], Any]],
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute zero-arg callables concurrently via threads.

        Ideal for GEE HTTP downloads because threads share the authenticated
        Earth Engine session within a process, avoiding per-process re-auth.

        Parameters
        ----------
        tasks      : {key: zero-arg callable}
        max_workers: defaults to len(tasks).

        Returns
        -------
        dict with the same keys mapping to each callable's return value.
        """
        n = max_workers or len(tasks)
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {k: pool.submit(fn) for k, fn in tasks.items()}
        return {k: f.result() for k, f in futures.items()}

    # ── xarray / raster helpers ──────────────────────────────────────────────

    @staticmethod
    def clip_raster_to_aoi(da_array, aoi_geojson: dict):
        """Clip a dask-backed xarray DataArray to an AOI polygon — lazy."""
        import rioxarray  # noqa: F401 — registers .rio accessor
        from shapely.geometry import shape

        geo_type = aoi_geojson.get("type")
        if geo_type == "Feature":
            geom_dict = aoi_geojson["geometry"]
        elif geo_type == "FeatureCollection":
            features = aoi_geojson.get("features", [])
            geom_dict = features[0]["geometry"] if features else aoi_geojson
        else:
            geom_dict = aoi_geojson
        geom = shape(geom_dict)
        return da_array.rio.clip([geom], crs="EPSG:4326", drop=True)

    @staticmethod
    def compute_with_progress(lazy_result):
        """Trigger Dask computation and return the materialised result."""
        return lazy_result.compute()
