"""
GEE authentication helpers for climate_change.

Usage contexts
--------------
Local / notebook
    Leave ``gee_project`` empty — the project is resolved from the ``.env``
    file, the ``GEE_PROJECT`` environment variable, or an interactive prompt.

Backend / API  (recommended pattern)
    1. At **user registration**, call ``validate_gee_project(project_id)``
       to verify the project is reachable before storing it in the database.
    2. On every **analysis request**, pass the stored project ID through
       ``run_analysis(..., gee_project=user.gee_project)``.
    3. Users are **never prompted again** — authentication is automatic from
       their stored project ID.

    Example FastAPI handler::

        @router.post("/analyse")
        async def analyse(body: AnalysisRequest, user: User = Depends(get_current_user)):
            # gee_project was verified at registration and stored in the DB.
            # Authentication happens automatically — the user is never asked again.
            return await run_analysis(
                module=body.module,
                aoi_geojson=body.aoi,
                start_date=body.start_date,
                end_date=body.end_date,
                country=body.country,
                gee_project=user.gee_project,
            )

Database security guidance
--------------------------
The GEE project ID stored in the database is **not a secret key** — it is a
Google Cloud project identifier.  However, it should be protected because:

* If leaked it could be used by someone with valid GEE credentials to bill
  charges against the project owner's quota.
* Users must never be able to read or impersonate another user's project ID.

Recommended storage practices:

1. **Encrypt the column at rest** — use your database's column-level encryption
   (PostgreSQL ``pgcrypto``, MySQL ``AES_ENCRYPT``) or a secrets manager
   (AWS KMS, GCP Cloud KMS, HashiCorp Vault) to encrypt the value before
   writing it.  The encryption key lives outside the database.

2. **Row-level security** — ensure the API never returns another user's
   ``gee_project`` field.  Add a database-level policy (e.g. PostgreSQL RLS)
   so queries are automatically scoped to the authenticated user.

3. **Encrypt the full database at rest** — all major cloud databases (RDS,
   Cloud SQL, Supabase) support transparent disk-level encryption; enable it.

4. **Audit access** — log every read of the ``gee_project`` column so
   suspicious bulk-reads are detectable.

What is NOT stored here:

* OAuth tokens and service-account keys stay on the server filesystem after
  ``earthengine authenticate``.  They are **never stored in the user DB**.
  Only the project ID (a non-secret identifier) is stored.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()

# Maps project_id → True once that project has been initialised in this process.
# Using a per-project key (not per-PID) handles multi-user API servers where
# different requests carry different users' project IDs in the same process.
_initialised_projects: set[str] = set()
_ANONYMOUS_KEY = "__anonymous__"  # sentinel for calls with no project


def _resolve_project(project: str, *, allow_prompt: bool = True) -> str:
    """
    Return the GEE Cloud project ID using this priority order:

    1. ``project`` argument passed by the caller
    2. ``GEE_PROJECT`` environment variable (or ``.env`` file)
    3. Interactive prompt — **only** when ``allow_prompt=True``

    In the API context ``allow_prompt`` is always ``False``, so a missing
    project raises ``ValueError`` immediately rather than blocking on
    ``input()``.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()

    if project:
        return project

    env_val = os.environ.get("GEE_PROJECT", "").strip()
    if env_val:
        return env_val

    if not allow_prompt:
        raise ValueError(
            "GEE project ID is required but was not provided.\n"
            "  • API/backend: the user's GEE project ID must be stored at "
            "registration and passed as gee_project= on every request.\n"
            "  • Local use: set GEE_PROJECT in your .env file or shell environment."
        )

    # Interactive fallback — local / notebook only
    try:
        entered = input(
            "\n  Google Earth Engine project ID not set.\n"
            "  Enter your GEE Cloud project ID (e.g. my-gee-project-123): "
        ).strip()
    except EOFError:
        entered = ""

    if entered:
        os.environ["GEE_PROJECT"] = entered
    return entered


def ensure_gee(project: str = "", *, allow_prompt: bool = False) -> None:
    """
    Authenticate and initialise GEE for the given project, once per project
    per process.

    Multi-user API behaviour
    ------------------------
    Each unique project ID is initialised independently.  Requests from
    User A and User B (different GEE projects) each trigger their own
    initialisation without interfering with each other.  Subsequent requests
    from the same user skip re-initialisation (idempotent).

    Project resolution order
    ------------------------
    1. ``project`` argument  (always used in API context)
    2. ``GEE_PROJECT`` environment variable / ``.env`` file  (local use)
    3. Interactive prompt (when ``allow_prompt=True``, local / notebook only)

    Parameters
    ----------
    project : str
        GEE Cloud project ID.  In API context this is always the value stored
        for the authenticated user — never empty.
    allow_prompt : bool
        ``False`` in API/server contexts; raises immediately on missing project.
    """
    import os

    with _lock:
        resolved = _resolve_project(project, allow_prompt=allow_prompt)
        cache_key = resolved or _ANONYMOUS_KEY

        if cache_key in _initialised_projects:
            return

        # Detect Dask worker context — workers cannot run interactive auth.
        in_worker = False
        try:
            from dask.distributed import get_worker

            get_worker()
            in_worker = True
        except (ImportError, ValueError):
            pass

        import ee

        kwargs: dict = {"project": resolved} if resolved else {}

        if in_worker:
            try:
                ee.Initialize(**kwargs)
            except Exception as exc:
                raise RuntimeError(
                    "GEE initialisation failed inside a Dask worker.\n"
                    "  Run  earthengine authenticate  in a terminal before\n"
                    "  starting the Dask cluster so credentials are on disk.\n"
                    f"Original error: {exc}"
                ) from exc
        else:
            try:
                from drought_monitoring.gee import authenticate

                authenticate(project=resolved or None, quiet=True)
            except ImportError:
                # drought_monitoring (ml extras) not installed — init directly
                ee.Initialize(**kwargs)

        _initialised_projects.add(cache_key)
        # Keep env in sync so sub-processes and libraries can read it.
        if resolved:
            os.environ["GEE_PROJECT"] = resolved


def startup_init_gee() -> None:
    """
    Authenticate and initialise GEE once at server startup using the
    organisation's project configured in ``GEE_PROJECT`` (env / .env file).

    Call this from the FastAPI lifespan inside ``asyncio.to_thread`` because
    ``ee.Initialize`` is synchronous.

    Logs a warning rather than raising if ``GEE_PROJECT`` is not set, so the
    server still starts in environments where GEE is not yet configured.
    """
    import logging
    import os

    from dotenv import load_dotenv

    load_dotenv()
    logger = logging.getLogger(__name__)

    project = os.environ.get("GEE_PROJECT", "").strip()
    if not project:
        logger.warning(
            "GEE_PROJECT is not set — Earth Engine analyses will fail until "
            "GEE_PROJECT is configured and the server is restarted."
        )
        return

    try:
        ensure_gee(project, allow_prompt=False)
        logger.info("Google Earth Engine initialised with project '%s'.", project)
    except Exception as exc:
        logger.error(
            "GEE initialisation failed at startup for project '%s': %s. "
            "Earth Engine analyses will be unavailable.",
            project,
            exc,
        )


def validate_gee_project(project: str) -> None:
    """
    Verify that a GEE project ID is valid and reachable.

    Call this **once at user registration** before storing the project ID in
    the database.  If this succeeds, ``ensure_gee`` will authenticate
    automatically on every subsequent request — the user is never asked again.

    Parameters
    ----------
    project : str
        The GEE Cloud project ID the user supplied during registration.

    Raises
    ------
    ValueError
        The project ID string is empty.
    RuntimeError
        GEE authentication or initialisation failed for this project.

    Example (FastAPI registration endpoint)::

        @router.post("/register")
        async def register(body: RegisterRequest):
            try:
                validate_gee_project(body.gee_project)
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            # Store encrypted in DB — user will never be prompted again.
            user = await create_user(body, gee_project=encrypt(body.gee_project))
            return {"id": user.id}
    """
    if not project or not project.strip():
        raise ValueError(
            "GEE project ID cannot be empty. "
            "Provide your Google Cloud project ID that has the Earth Engine API enabled."
        )
    try:
        ensure_gee(project.strip(), allow_prompt=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not authenticate with Google Earth Engine "
            f"using project '{project}'.\n"
            "Make sure:\n"
            "  1. The project exists in Google Cloud Console.\n"
            "  2. The Earth Engine API is enabled for this project.\n"
            "  3. Your service-account key or OAuth credentials are valid.\n"
            f"Details: {exc}"
        ) from exc
