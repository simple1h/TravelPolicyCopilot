from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings


def apply_migrations(
    *,
    dsn: str | None = None,
    migrations_dir: str = "migrations",
) -> list[str]:
    psycopg = __import__("psycopg")
    resolved_dsn = dsn or settings.postgres_dsn
    migration_root = ROOT / migrations_dir
    applied: list[str] = []

    with psycopg.connect(resolved_dsn, connect_timeout=int(settings.request_timeout_seconds)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.commit()

            for file_path in sorted(migration_root.glob("*.sql")):
                version = file_path.name
                cursor.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
                if cursor.fetchone():
                    continue
                sql = file_path.read_text(encoding="utf-8")
                cursor.execute(sql)
                cursor.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (version,))
                applied.append(version)
            connection.commit()
    return applied


def main() -> None:
    applied = apply_migrations()
    print({"applied_migrations": applied})


if __name__ == "__main__":
    main()
