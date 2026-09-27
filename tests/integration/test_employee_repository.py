from __future__ import annotations

import json

import pytest

from src.config import settings
from src.persistence.repository import PostgresEmployeeRepository


def test_postgres_employee_repository_reads_profile():
    psycopg = pytest.importorskip("psycopg")

    try:
        with psycopg.connect(settings.postgres_dsn, connect_timeout=2) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS employees (
                        employee_id TEXT PRIMARY KEY,
                        employee_category TEXT NOT NULL,
                        cost_center TEXT NOT NULL,
                        grade TEXT NOT NULL,
                        home_city TEXT NOT NULL,
                        policy_tier TEXT NOT NULL,
                        max_hotel_per_night INT NOT NULL,
                        max_flight_class TEXT NOT NULL,
                        requires_preapproval_over_amount INT NOT NULL,
                        accessibility_requirements JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO employees (
                        employee_id,
                        employee_category,
                        cost_center,
                        grade,
                        home_city,
                        policy_tier,
                        max_hotel_per_night,
                        max_flight_class,
                        requires_preapproval_over_amount,
                        accessibility_requirements
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (employee_id)
                    DO UPDATE SET
                        employee_category = EXCLUDED.employee_category,
                        cost_center = EXCLUDED.cost_center,
                        grade = EXCLUDED.grade,
                        home_city = EXCLUDED.home_city,
                        policy_tier = EXCLUDED.policy_tier,
                        max_hotel_per_night = EXCLUDED.max_hotel_per_night,
                        max_flight_class = EXCLUDED.max_flight_class,
                        requires_preapproval_over_amount = EXCLUDED.requires_preapproval_over_amount,
                        accessibility_requirements = EXCLUDED.accessibility_requirements
                    """,
                    (
                        "EMP-T4-001",
                        "manager",
                        "CC-T4",
                        "M3",
                        "Kazan",
                        "manager",
                        12000,
                        "business",
                        95000,
                        json.dumps(["priority_boarding"]),
                    ),
                )
            connection.commit()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL is not available for integration test: {exc}")

    repository = PostgresEmployeeRepository(settings.postgres_dsn, timeout_seconds=2)
    profile = repository.get_by_id("EMP-T4-001")

    assert profile is not None
    assert profile.employee_id == "EMP-T4-001"
    assert profile.employee_category == "manager"
    assert profile.max_hotel_per_night == 12000
    assert profile.accessibility_requirements == ["priority_boarding"]
