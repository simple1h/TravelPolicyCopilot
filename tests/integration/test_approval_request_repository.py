from __future__ import annotations

import json
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

from src.config import settings
from src.domain.models import ApprovalRequest, BudgetCheckResult, PolicyCheckResult, TripRequest
from src.persistence.repository import PostgresApprovalRequestRepository


def test_postgres_approval_request_create_is_idempotent():
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
                    CREATE TABLE IF NOT EXISTS approval_requests (
                        request_id TEXT PRIMARY KEY,
                        employee_id TEXT NOT NULL REFERENCES employees(employee_id),
                        trip_request JSONB NOT NULL,
                        selected_option_id TEXT NOT NULL,
                        total_price INT NOT NULL,
                        budget_result JSONB NOT NULL,
                        policy_results JSONB NOT NULL,
                        citations JSONB NOT NULL,
                        status TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        idempotency_key TEXT NOT NULL UNIQUE
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
                    ON CONFLICT (employee_id) DO NOTHING
                    """,
                    (
                        "EMP-T5-001",
                        "standard",
                        "CC-T5",
                        "G6",
                        "Kazan",
                        "default",
                        7000,
                        "economy",
                        45000,
                        json.dumps([]),
                    ),
                )
            connection.commit()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL is not available for integration test: {exc}")

    repository = PostgresApprovalRequestRepository(settings.postgres_dsn, timeout_seconds=2)
    idem_key = f"idem-{uuid4().hex}"

    first = ApprovalRequest(
        request_id=f"REQ-{uuid4().hex[:8].upper()}",
        employee_id="EMP-T5-001",
        trip_request=TripRequest(
            origin_city="Казань",
            destination_city="Москва",
            departure_date=date(2026, 9, 18),
            return_date=date(2026, 9, 19),
        ),
        selected_option_id="OPT-T5-001",
        total_price=42000,
        budget_result=BudgetCheckResult(
            verdict="within_limit",
            total_price=42000,
            threshold=45000,
            overage=0,
        ),
        policy_results=[
            PolicyCheckResult(
                rule_id="TRANSPORT_CLASS_LIMIT",
                verdict="pass",
                message="Fare class is allowed.",
                citations=[],
            )
        ],
        citations=[],
        status="draft_created",
        created_at=datetime.now(UTC),
        idempotency_key=idem_key,
    )
    second = first.model_copy(update={"request_id": f"REQ-{uuid4().hex[:8].upper()}"})

    stored_first = repository.create_idempotent(first)
    stored_second = repository.create_idempotent(second)

    assert stored_first.request_id == stored_second.request_id
    assert stored_first.idempotency_key == idem_key
    assert stored_first.employee_id == "EMP-T5-001"
