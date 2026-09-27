from __future__ import annotations

import pytest

from scripts.migrate_postgres import apply_migrations
from scripts.seed_employees import seed_employees
from src.config import settings


def test_migrations_and_seed_create_required_tables_and_profiles():
    psycopg = pytest.importorskip("psycopg")

    try:
        applied = apply_migrations()
        seeded = seed_employees()
    except Exception as exc:  # pragma: no cover - env/runtime dependent
        pytest.skip(f"PostgreSQL is not available for bootstrap integration test: {exc}")

    assert isinstance(applied, list)
    assert seeded >= 5

    with psycopg.connect(settings.postgres_dsn, connect_timeout=2) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM employees")
            employees_count = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('employees', 'approval_requests', 'schema_migrations')
                """
            )
            tables_count = int(cursor.fetchone()[0])

    assert employees_count >= 5
    assert tables_count == 3
