from __future__ import annotations

import logging
import time

import requests

_log = logging.getLogger(__name__)

# Only retry failures that are plausibly transient — a dropped/reset
# connection or a timeout. HTTP error statuses (4xx/5xx) are a real
# rejection (e.g. Earth Engine explicitly refusing a malformed request)
# that retrying won't fix, so those are left to the caller's own
# resp.raise_for_status() handling, same as before this helper existed.
_RETRYABLE_EXCEPTIONS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


def get_with_retry(
    url: str,
    timeout: int = 600,
    attempts: int = 3,
    backoff: float = 0.75,
    **kwargs: object,
) -> requests.Response:
    """
    GET a URL, retrying on a transient connection failure (dropped/reset
    connection, timeout) with short, bounded backoff.

    Confirmed live: a Google Earth Engine download URL failed mid-fetch
    with `ConnectionError: RemoteDisconnected` under server load — a
    short retry would very likely have succeeded. `_download_band()`-style
    helpers across flood/food_security/disease/land_degradation/core all
    hit this same class of URL and previously had no retry at all.

    Backoff is kept short and linear (not exponential) because these calls
    run inside small ThreadPoolExecutor pools shared with sibling downloads
    (see core.dask_engine.DaskEngine.run_io_parallel, max_workers=2-4) — a
    long sleep here would starve the pool rather than just this one caller.
    """
    for attempt in range(1, attempts + 1):
        try:
            return requests.get(url, timeout=timeout, **kwargs)  # type: ignore[arg-type]
        except _RETRYABLE_EXCEPTIONS as exc:
            if attempt == attempts:
                raise
            _log.warning(
                "GET %s failed (attempt %d/%d): %s — retrying in %.2fs",
                url,
                attempt,
                attempts,
                exc,
                backoff * attempt,
            )
            time.sleep(backoff * attempt)
    raise AssertionError("unreachable")  # loop above always returns or raises
