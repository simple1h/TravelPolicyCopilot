from __future__ import annotations

from collections.abc import Iterable
import json
from threading import Lock

from src.config import is_strict_mode, settings
from src.domain.models import ApprovalRequest, EmployeeProfile


class EmployeeRepository:
    def get_by_id(self, employee_id: str) -> EmployeeProfile | None:
        """DB adapter placeholder."""
        return None


class PostgresEmployeeRepository(EmployeeRepository):
    def __init__(self, dsn: str, *, timeout_seconds: float = 5.0) -> None:
        self._dsn = dsn
        self._timeout_seconds = timeout_seconds

    def _connect(self):
        try:
            import psycopg
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("psycopg is required for PostgreSQL repository.") from exc
        return psycopg.connect(self._dsn, connect_timeout=int(self._timeout_seconds))

    def get_by_id(self, employee_id: str) -> EmployeeProfile | None:
        query = """
            SELECT
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
            FROM employees
            WHERE employee_id = %s
            LIMIT 1
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (employee_id,))
                row = cursor.fetchone()

        if row is None:
            return None

        accessibility_requirements = row[9]
        if isinstance(accessibility_requirements, str):
            try:
                accessibility_requirements = json.loads(accessibility_requirements)
            except json.JSONDecodeError:
                accessibility_requirements = []

        return EmployeeProfile(
            employee_id=row[0],
            employee_category=row[1],
            cost_center=row[2],
            grade=row[3],
            home_city=row[4],
            policy_tier=row[5],
            max_hotel_per_night=int(row[6]),
            max_flight_class=row[7],
            requires_preapproval_over_amount=int(row[8]),
            accessibility_requirements=accessibility_requirements or [],
        )


class ApprovalRequestRepository:
    def create_idempotent(self, request: ApprovalRequest) -> ApprovalRequest:
        """Insert with idempotency_key unique constraint."""
        return request

    def get_by_request_id(self, request_id: str) -> ApprovalRequest | None:
        return None

    def list_for_employee(self, employee_id: str) -> Iterable[ApprovalRequest]:
        return []


class PostgresApprovalRequestRepository(ApprovalRequestRepository):
    def __init__(self, dsn: str, *, timeout_seconds: float = 5.0) -> None:
        self._dsn = dsn
        self._timeout_seconds = timeout_seconds

    def _connect(self):
        try:
            import psycopg
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("psycopg is required for PostgreSQL repository.") from exc
        return psycopg.connect(self._dsn, connect_timeout=int(self._timeout_seconds))

    @staticmethod
    def _to_approval_request(row) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=row[0],
            employee_id=row[1],
            trip_request=row[2],
            selected_option_id=row[3],
            total_price=int(row[4]),
            budget_result=row[5],
            policy_results=row[6],
            citations=row[7],
            status=row[8],
            created_at=row[9],
            idempotency_key=row[10],
        )

    def create_idempotent(self, request: ApprovalRequest) -> ApprovalRequest:
        insert_query = """
            INSERT INTO approval_requests (
                request_id,
                employee_id,
                trip_request,
                selected_option_id,
                total_price,
                budget_result,
                policy_results,
                citations,
                status,
                created_at,
                idempotency_key
            )
            VALUES (
                %s,
                %s,
                %s::jsonb,
                %s,
                %s,
                %s::jsonb,
                %s::jsonb,
                %s::jsonb,
                %s,
                %s,
                %s
            )
            ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING
                request_id,
                employee_id,
                trip_request,
                selected_option_id,
                total_price,
                budget_result,
                policy_results,
                citations,
                status,
                created_at,
                idempotency_key
        """
        select_by_idempotency_query = """
            SELECT
                request_id,
                employee_id,
                trip_request,
                selected_option_id,
                total_price,
                budget_result,
                policy_results,
                citations,
                status,
                created_at,
                idempotency_key
            FROM approval_requests
            WHERE idempotency_key = %s
            LIMIT 1
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    insert_query,
                    (
                        request.request_id,
                        request.employee_id,
                        json.dumps(request.trip_request.model_dump(mode="json")),
                        request.selected_option_id,
                        request.total_price,
                        json.dumps(request.budget_result.model_dump(mode="json")),
                        json.dumps([item.model_dump(mode="json") for item in request.policy_results]),
                        json.dumps([item.model_dump(mode="json") for item in request.citations]),
                        request.status,
                        request.created_at,
                        request.idempotency_key,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(select_by_idempotency_query, (request.idempotency_key,))
                    row = cursor.fetchone()
            connection.commit()

        if row is None:
            raise RuntimeError("Failed to insert or fetch idempotent approval request.")

        return self._to_approval_request(row)

    def get_by_request_id(self, request_id: str) -> ApprovalRequest | None:
        query = """
            SELECT
                request_id,
                employee_id,
                trip_request,
                selected_option_id,
                total_price,
                budget_result,
                policy_results,
                citations,
                status,
                created_at,
                idempotency_key
            FROM approval_requests
            WHERE request_id = %s
            LIMIT 1
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (request_id,))
                row = cursor.fetchone()
        if row is None:
            return None
        return self._to_approval_request(row)

    def list_for_employee(self, employee_id: str) -> Iterable[ApprovalRequest]:
        query = """
            SELECT
                request_id,
                employee_id,
                trip_request,
                selected_option_id,
                total_price,
                budget_result,
                policy_results,
                citations,
                status,
                created_at,
                idempotency_key
            FROM approval_requests
            WHERE employee_id = %s
            ORDER BY created_at DESC
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (employee_id,))
                rows = cursor.fetchall()
        return [self._to_approval_request(row) for row in rows]


class InMemoryApprovalRequestRepository(ApprovalRequestRepository):
    def __init__(self) -> None:
        self._by_idempotency_key: dict[str, ApprovalRequest] = {}
        self._by_request_id: dict[str, ApprovalRequest] = {}
        self._lock = Lock()

    def create_idempotent(self, request: ApprovalRequest) -> ApprovalRequest:
        with self._lock:
            existing = self._by_idempotency_key.get(request.idempotency_key)
            if existing is not None:
                return existing

            self._by_idempotency_key[request.idempotency_key] = request
            self._by_request_id[request.request_id] = request
            return request

    def get_by_request_id(self, request_id: str) -> ApprovalRequest | None:
        return self._by_request_id.get(request_id)

    def list_for_employee(self, employee_id: str) -> Iterable[ApprovalRequest]:
        return [item for item in self._by_request_id.values() if item.employee_id == employee_id]


class FallbackApprovalRequestRepository(ApprovalRequestRepository):
    def __init__(self, primary: ApprovalRequestRepository, fallback: ApprovalRequestRepository) -> None:
        self._primary = primary
        self._fallback = fallback

    def create_idempotent(self, request: ApprovalRequest) -> ApprovalRequest:
        try:
            return self._primary.create_idempotent(request)
        except Exception:
            return self._fallback.create_idempotent(request)

    def get_by_request_id(self, request_id: str) -> ApprovalRequest | None:
        try:
            result = self._primary.get_by_request_id(request_id)
            if result is not None:
                return result
        except Exception:
            pass
        return self._fallback.get_by_request_id(request_id)

    def list_for_employee(self, employee_id: str) -> Iterable[ApprovalRequest]:
        try:
            primary_items = list(self._primary.list_for_employee(employee_id))
            if primary_items:
                return primary_items
        except Exception:
            pass
        return self._fallback.list_for_employee(employee_id)


approval_request_repository = FallbackApprovalRequestRepository(
    primary=PostgresApprovalRequestRepository(
        settings.postgres_dsn,
        timeout_seconds=min(settings.request_timeout_seconds, 10.0),
    ),
    fallback=InMemoryApprovalRequestRepository(),
)
if is_strict_mode():
    approval_request_repository = PostgresApprovalRequestRepository(
        settings.postgres_dsn,
        timeout_seconds=min(settings.request_timeout_seconds, 10.0),
    )
employee_repository = PostgresEmployeeRepository(
    settings.postgres_dsn,
    timeout_seconds=min(settings.request_timeout_seconds, 10.0),
)
