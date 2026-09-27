from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings


def seed_employees(
    *,
    dsn: str | None = None,
    dataset_path: str = "data/employees/seed_employees.json",
) -> int:
    psycopg = __import__("psycopg")
    resolved_dsn = dsn or settings.postgres_dsn
    rows = json.loads((ROOT / dataset_path).read_text(encoding="utf-8"))

    with psycopg.connect(resolved_dsn, connect_timeout=int(settings.request_timeout_seconds)) as connection:
        with connection.cursor() as cursor:
            for item in rows:
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
                        item["employee_id"],
                        item["employee_category"],
                        item["cost_center"],
                        item["grade"],
                        item["home_city"],
                        item["policy_tier"],
                        int(item["max_hotel_per_night"]),
                        item["max_flight_class"],
                        int(item["requires_preapproval_over_amount"]),
                        json.dumps(item.get("accessibility_requirements", [])),
                    ),
                )
            connection.commit()
    return len(rows)


def main() -> None:
    count = seed_employees()
    print({"seeded_employees": count})


if __name__ == "__main__":
    main()
