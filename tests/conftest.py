from pathlib import Path

import pytest

from app.registry import Registry
from app.store import VectorStore

POLICIES_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"


def make_doc(company: str = "acme", version: int = 1, sections: dict | None = None,
             title: str = "Acme Policy") -> str:
    """Build a small policy document in the same format as data/policies/*.md."""
    sections = sections or {"1.1": ("Scope", "This policy covers cabs.")}
    body = "\n\n".join(f"## {num} {heading}\n{text}" for num, (heading, text) in sections.items())
    return (
        f"---\ncompany_id: {company}\ntitle: {title}\nversion: {version}\n"
        f"effective_date: 2026-01-01\n---\n# {title}\n\n{body}\n"
    )


@pytest.fixture(scope="session")
def seeded(tmp_path_factory) -> tuple[Registry, VectorStore]:
    """The real seed data, loaded once per test run. Only for tests that just read."""
    from app.ingestion import seed_policies

    root = tmp_path_factory.mktemp("seeded")
    registry, store = Registry(root / "r.sqlite3"), VectorStore(root / "chroma")
    seed_policies(POLICIES_DIR, registry=registry, store=store)
    return registry, store


@pytest.fixture
def registry(tmp_path) -> Registry:
    return Registry(tmp_path / "registry.sqlite3")


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")
