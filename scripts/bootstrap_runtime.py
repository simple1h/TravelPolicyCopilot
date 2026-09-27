from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.migrate_postgres import apply_migrations
from scripts.seed_employees import seed_employees
from src.config import settings
from src.rag.ingestion import ingest_policies


def _wait_for_postgres(*, timeout_seconds: float = 60.0) -> None:
    psycopg = __import__("psycopg")
    deadline = time.time() + timeout_seconds
    last_error = "unknown"
    while time.time() < deadline:
        try:
            with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
                return
        except Exception as exc:  # pragma: no cover - runtime depends on env
            last_error = str(exc)
            time.sleep(1.5)
    raise RuntimeError(f"PostgreSQL is not ready in time: {last_error}")


def _wait_for_http(url: str, *, timeout_seconds: float = 60.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error = "unknown"
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except Exception as exc:  # pragma: no cover - runtime depends on env
            last_error = str(exc)
        time.sleep(1.5)
    raise RuntimeError(f"Endpoint is not ready in time {url}: {last_error}")


def bootstrap_runtime() -> dict[str, object]:
    _wait_for_postgres()
    _wait_for_http(f"{settings.qdrant_url.rstrip('/')}/collections")
    _wait_for_http(f"{settings.mock_travel_api_url.rstrip('/')}/health")

    applied_migrations = apply_migrations()
    seeded_employees = seed_employees()
    ingestion = ingest_policies(with_qdrant=True)
    if not ingestion.qdrant_upserted:
        raise RuntimeError(f"Policy ingestion to Qdrant failed: {ingestion.qdrant_error}")

    return {
        "applied_migrations": applied_migrations,
        "seeded_employees": seeded_employees,
        "ingestion": {
            "chunks_count": ingestion.chunks_count,
            "output_path": ingestion.output_path,
            "qdrant_upserted": ingestion.qdrant_upserted,
        },
    }


def main() -> None:
    result = bootstrap_runtime()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
