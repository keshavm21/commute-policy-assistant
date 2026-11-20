"""SQLite registry of policy versions: the single source of truth for which version is active.

Each ingested version lives in its own Chroma collection. This table points at them.
Switching the active version is one SQLite transaction, which is what makes an
update atomic: readers see either the old version or the new one, never a mix.
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id      TEXT    NOT NULL,
    version         INTEGER NOT NULL,
    title           TEXT    NOT NULL,
    effective_date  TEXT    NOT NULL,
    collection_name TEXT    NOT NULL UNIQUE,
    content_sha256  TEXT    NOT NULL,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL CHECK (status IN ('building', 'active', 'retired', 'failed')),
    error           TEXT,
    created_at      TEXT    NOT NULL
);
-- The database itself guarantees at most one active version per company.
CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_company
    ON policy_versions (company_id) WHERE status = 'active';
-- A version number can be used once per company (failed attempts may be retried).
CREATE UNIQUE INDEX IF NOT EXISTS one_row_per_version
    ON policy_versions (company_id, version) WHERE status != 'failed';
"""


@dataclass(frozen=True)
class PolicyVersion:
    id: int
    company_id: str
    version: int
    title: str
    effective_date: str
    collection_name: str
    content_sha256: str
    chunk_count: int
    status: str


class Registry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per call keeps this safe across FastAPI threads.
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _one(self, sql: str, params: tuple) -> PolicyVersion | None:
        with closing(self._connect()) as conn:
            row = conn.execute(sql, params).fetchone()
        return _to_version(row) if row else None

    # ---- reads -------------------------------------------------------------

    def get_active(self, company_id: str) -> PolicyVersion | None:
        return self._one(
            "SELECT * FROM policy_versions WHERE company_id = ? AND status = 'active'",
            (company_id,),
        )

    def latest_version_number(self, company_id: str) -> int:
        """Highest version ever made live (active or retired) for this company, 0 if none."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT MAX(version) FROM policy_versions "
                "WHERE company_id = ? AND status IN ('active', 'retired')",
                (company_id,),
            ).fetchone()
        return row[0] or 0

    def find_live_by_hash(self, company_id: str, sha256: str) -> PolicyVersion | None:
        return self._one(
            "SELECT * FROM policy_versions WHERE company_id = ? AND content_sha256 = ? "
            "AND status IN ('active', 'retired')",
            (company_id, sha256),
        )

    def list_versions(self, company_id: str) -> list[PolicyVersion]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM policy_versions WHERE company_id = ? ORDER BY id", (company_id,)
            ).fetchall()
        return [_to_version(r) for r in rows]

    def list_building(self) -> list[PolicyVersion]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM policy_versions WHERE status = 'building'").fetchall()
        return [_to_version(r) for r in rows]

    # ---- writes ------------------------------------------------------------

    def start_build(
        self, *, company_id: str, version: int, title: str, effective_date: str,
        collection_name: str, sha256: str,
    ) -> int:
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO policy_versions (company_id, version, title, effective_date, "
                "collection_name, content_sha256, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'building', ?)",
                (company_id, version, title, effective_date, collection_name, sha256, _now()),
            )
            return cur.lastrowid

    def mark_failed(self, row_id: int, error: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE policy_versions SET status = 'failed', error = ? WHERE id = ?",
                (error[:500], row_id),
            )

    def activate(self, row_id: int, chunk_count: int) -> None:
        """Retire the current active version and activate `row_id`, in ONE transaction.

        If anything here fails, SQLite rolls back and the old version stays active.
        """
        with closing(self._connect()) as conn:
            conn.isolation_level = None  # manage the transaction explicitly
            conn.execute("BEGIN IMMEDIATE")  # take the write lock before reading
            try:
                row = conn.execute(
                    "SELECT company_id FROM policy_versions WHERE id = ? AND status = 'building'",
                    (row_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(f"version row {row_id} is not in 'building' state")
                conn.execute(
                    "UPDATE policy_versions SET status = 'retired' "
                    "WHERE company_id = ? AND status = 'active'",
                    (row["company_id"],),
                )
                conn.execute(
                    "UPDATE policy_versions SET status = 'active', chunk_count = ? WHERE id = ?",
                    (chunk_count, row_id),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise


def _to_version(row: sqlite3.Row) -> PolicyVersion:
    return PolicyVersion(
        id=row["id"], company_id=row["company_id"], version=row["version"], title=row["title"],
        effective_date=row["effective_date"], collection_name=row["collection_name"],
        content_sha256=row["content_sha256"], chunk_count=row["chunk_count"], status=row["status"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
