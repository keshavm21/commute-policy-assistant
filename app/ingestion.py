"""Policy ingestion: parse -> validate -> chunk -> build new collection -> atomic switch.

Order matters. All validation happens before anything is written. The new version is
built in its own collection, and the registry pointer flips to it only after the
collection is complete and verified. Any failure before the flip leaves the old version
active and untouched, and the half-built collection is deleted.
"""

import hashlib
import logging
import re
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from app.registry import Registry
from app.store import Chunk, VectorStore

log = logging.getLogger(__name__)

COMPANY_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
SECTION_RE = re.compile(r"^##\s+(\d+(?:\.\d+)*)\s+(\S.*)$")  # "## 4.2 Security escort"
MAX_DOC_CHARS = 200_000
MAX_CHUNK_CHARS = 1_200  # longer sections are split by paragraph, keeping the section label

# Text that talks to an AI instead of to employees. A real policy rule never does this,
# so such a section is quarantined: kept out of the index and reported to the admin.
# This only catches known phrasings; the prompt rule in prompts.py is the second layer.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in (
        r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions?|rules|prompt)\b",
        r"\b(ai|automated|virtual)\s+(assistant|model|agent)s?\b",
        r"\b(language model|llm|chatbot)s?\b",
        r"\bsystem\s+(notice|prompt|message|instructions?)\b",
        r"\byou are now\b",
    )
]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


# Ingestion is rare; serialising it removes a whole class of race conditions.
_INGEST_LOCK = threading.Lock()


class IngestionError(Exception):
    """The document was rejected. Nothing was changed."""


@dataclass(frozen=True)
class ParsedPolicy:
    company_id: str
    title: str
    version: int
    effective_date: str
    sections: list[tuple[str, str, str]]  # (section number, heading, body)
    sha256: str


@dataclass(frozen=True)
class IngestResult:
    status: str  # "ingested" | "unchanged"
    company_id: str
    version: int
    chunk_count: int
    quarantined_sections: tuple[str, ...] = ()  # left out because they address an AI


# ---------------------------------------------------------------------------
# Parsing and validation (pure functions, no side effects)
# ---------------------------------------------------------------------------

def parse_policy(text: str) -> ParsedPolicy:
    if not text or not text.strip():
        raise IngestionError("document is empty")
    if len(text) > MAX_DOC_CHARS:
        raise IngestionError(f"document is larger than {MAX_DOC_CHARS} characters")

    meta, body = _split_front_matter(text)
    company_id = meta.get("company_id")
    title = meta.get("title")
    version = meta.get("version")
    effective_date = meta.get("effective_date")

    if not isinstance(company_id, str) or not COMPANY_ID_RE.match(company_id):
        raise IngestionError("front matter 'company_id' missing or invalid (lowercase slug)")
    if not isinstance(title, str) or not title.strip():
        raise IngestionError("front matter 'title' missing")
    # bool is a subclass of int in Python, so exclude it explicitly
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise IngestionError("front matter 'version' must be a positive integer")
    if effective_date is None:
        raise IngestionError("front matter 'effective_date' missing")

    sections = _parse_sections(body)
    return ParsedPolicy(
        company_id=company_id, title=title.strip(), version=version,
        effective_date=str(effective_date), sections=sections,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _split_front_matter(text: str) -> tuple[dict, str]:
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        raise IngestionError("document must start with a '---' front matter block")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise IngestionError("front matter block is not closed with '---'") from None
    try:
        meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise IngestionError(f"front matter is not valid YAML: {exc}") from None
    if not isinstance(meta, dict):
        raise IngestionError("front matter must be a mapping")
    return meta, "\n".join(lines[end + 1:])


def _parse_sections(body: str) -> list[tuple[str, str, str]]:
    sections: list[tuple[str, str, list[str]]] = []
    for line in body.splitlines():
        match = SECTION_RE.match(line.strip())
        if match:
            sections.append((match.group(1), match.group(2).strip(), []))
        elif sections:
            sections[-1][2].append(line)
        elif line.strip() and not line.startswith("# "):
            # Only a "# Title" line may come before the first section; anything else
            # would be text we cannot cite, so reject instead of silently dropping it.
            raise IngestionError(f"text before the first '## <number> <heading>': {line[:60]!r}")

    if not sections:
        raise IngestionError("no sections found (expected lines like '## 4.2 Heading')")

    seen: set[str] = set()
    result = []
    for number, heading, body_lines in sections:
        if number in seen:
            raise IngestionError(f"duplicate section number {number}")
        seen.add(number)
        text = "\n".join(body_lines).strip()
        if not text:
            raise IngestionError(f"section {number} is empty")
        result.append((number, heading, text))
    return result


def split_quarantined(policy: ParsedPolicy) -> tuple[ParsedPolicy, tuple[str, ...]]:
    """Remove sections that look like prompt injection; return (clean policy, their numbers)."""
    kept = [s for s in policy.sections if not looks_like_injection(f"{s[1]}\n{s[2]}")]
    removed = tuple(s[0] for s in policy.sections if s not in kept)
    if not kept:
        raise IngestionError("every section looks like instructions to an AI; nothing to index")
    return replace(policy, sections=kept), removed


def chunk_policy(policy: ParsedPolicy) -> list[Chunk]:
    """One chunk per section. A long section is split on blank lines, same section label."""
    chunks = []
    for number, heading, body in policy.sections:
        parts = _split_long(body) if len(body) > MAX_CHUNK_CHARS else [body]
        for i, part in enumerate(parts, start=1):
            chunk_id = number if len(parts) == 1 else f"{number}#p{i}"
            chunks.append(Chunk(
                chunk_id=chunk_id, text=f"{heading}\n{part}", company_id=policy.company_id,
                version=policy.version, doc_title=policy.title,
                effective_date=policy.effective_date, section=number, heading=heading, body=part,
            ))
    return chunks


def _split_long(body: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > MAX_CHUNK_CHARS:
            parts.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        parts.append(current)
    return parts


# ---------------------------------------------------------------------------
# Ingestion with side effects
# ---------------------------------------------------------------------------

def ingest_policy(
    text: str, *, registry: Registry, store: VectorStore, expected_company: str | None = None,
) -> IngestResult:
    policy = parse_policy(text)  # step 1: validate; raises IngestionError, nothing changed
    if expected_company is not None and policy.company_id != expected_company:
        raise IngestionError(
            f"document is for company '{policy.company_id}', not '{expected_company}'"
        )
    clean, quarantined = split_quarantined(policy)
    if quarantined:
        log.warning("%s v%s: quarantined section(s) %s (text addressed to an AI)",
                    policy.company_id, policy.version, ", ".join(quarantined))
    chunks = chunk_policy(clean)

    with _INGEST_LOCK:
        existing = registry.find_live_by_hash(policy.company_id, policy.sha256)
        if existing:  # identical file already ingested: idempotent no-op
            return IngestResult("unchanged", existing.company_id, existing.version,
                                existing.chunk_count)

        latest = registry.latest_version_number(policy.company_id)
        if policy.version <= latest:
            raise IngestionError(
                f"version {policy.version} is not newer than the latest version {latest}"
            )

        # step 2: record the attempt; step 3: build a NEW collection the readers can't see yet
        collection = f"{policy.company_id}__v{policy.version}__{uuid.uuid4().hex[:8]}"
        row_id = registry.start_build(
            company_id=policy.company_id, version=policy.version, title=policy.title,
            effective_date=policy.effective_date, collection_name=collection,
            sha256=policy.sha256,
        )
        try:
            stored = store.create_collection(collection, chunks)
            if stored != len(chunks):
                raise RuntimeError(f"stored {stored} chunks, expected {len(chunks)}")
            # step 4: the only moment the active version changes, one SQLite transaction
            registry.activate(row_id, chunk_count=stored)
        except Exception as exc:
            # step 5: roll back. The old version was never touched, so it keeps answering.
            log.exception("ingestion of %s v%s failed", policy.company_id, policy.version)
            store.delete_collection(collection)
            registry.mark_failed(row_id, repr(exc))
            raise

    log.info("activated %s v%s (%d chunks)", policy.company_id, policy.version, stored)
    return IngestResult("ingested", policy.company_id, policy.version, stored, quarantined)


def cleanup_interrupted_builds(registry: Registry, store: VectorStore) -> int:
    """On startup: anything still 'building' was interrupted by a crash. Remove it."""
    stale = registry.list_building()
    for row in stale:
        store.delete_collection(row.collection_name)
        registry.mark_failed(row.id, "interrupted before activation (found at startup)")
    return len(stale)


def seed_policies(policies_dir: Path, *, registry: Registry, store: VectorStore) -> None:
    """Ingest every policy file found on disk, oldest version first. Safe to rerun."""
    parsed = []
    for path in sorted(policies_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        try:
            p = parse_policy(text)
        except IngestionError as exc:
            log.error("skipping %s: %s", path.name, exc)
            continue
        parsed.append((p.company_id, p.version, path.name, text, p.sha256))

    for company_id, _version, name, text, sha256 in sorted(parsed):
        if registry.find_live_by_hash(company_id, sha256):
            continue  # already ingested on a previous start
        try:
            ingest_policy(text, registry=registry, store=store)
        except Exception as exc:  # one bad file must not stop the other companies loading
            log.error("could not ingest %s: %s", name, exc)
