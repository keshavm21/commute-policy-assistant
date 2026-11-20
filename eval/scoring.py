"""Deterministic scoring for one eval case. No LLM judge: same input -> same score, free.

A case passes only if EVERY check below passes:
  isolation   - every retrieved chunk belongs to the asker's company      (always)
  freshness   - every retrieved chunk is from that company's active version (always)
  status      - status is the expected one (or one of the allowed ones)
  citation    - if answered and sections are given: a cited section is an expected one
  fact        - if answered and must_contain_any is given: the answer contains one of them
  forbidden   - the answer contains none of must_not_contain               (always)
Text matching is case-insensitive whole-phrase matching on normalised text (see `contains`).
"""

import re
import unicodedata

_DASHES = re.compile(r"[‐-―−]")  # hyphen/dash look-alikes -> "-"


def normalise(text: str) -> str:
    """Lower-case, fold Unicode look-alikes, collapse whitespace.

    Real LLMs write "8:00 PM" (narrow no-break space) or "same‑day" (non-breaking
    hyphen). Without this, correct answers fail a plain substring match.
    """
    text = unicodedata.normalize("NFKC", text)  # e.g. U+202F / U+00A0 -> plain space
    text = _DASHES.sub("-", text)
    return " ".join(text.lower().split())


def contains(answer: str, phrase: str) -> bool:
    """Whole-phrase match on normalised text: "4 hours" must not match inside "24 hours",
    and "1 week" must not match inside "5.1 weekend"."""
    pattern = r"(?<![\w.,])" + re.escape(normalise(phrase)) + r"(?![\w])"
    return re.search(pattern, normalise(answer)) is not None


def score_case(expected: dict, actual: dict, company: str, active_version: int) -> list[str]:
    """Return the list of failed checks (empty list = pass)."""
    failed = []
    retrieved = actual["retrieved"]
    answer = actual["answer"]

    if any(r["company"] != company for r in retrieved):
        failed.append("isolation: retrieved another company's chunk")
    if any(r["version"] != active_version for r in retrieved):
        failed.append("freshness: retrieved a non-active version")

    allowed = expected["status"] if isinstance(expected["status"], list) else [expected["status"]]
    if actual["status"] not in allowed:
        failed.append(f"status: got {actual['status']}, expected {'/'.join(allowed)}")

    if actual["status"] == "answered":
        sections = expected.get("sections") or []
        if sections and not set(actual["cited_sections"]) & set(sections):
            failed.append(f"citation: cited {actual['cited_sections']}, expected one of {sections}")
        options = expected.get("must_contain_any") or []
        if options and not any(contains(answer, o) for o in options):
            failed.append(f"fact: answer has none of {options}")

    hits = [w for w in expected.get("must_not_contain") or [] if contains(answer, w)]
    if hits:
        failed.append(f"forbidden: answer contains {hits}")
    return failed


def retrieval_hit(expected: dict, actual: dict) -> bool | None:
    """Was an expected section among the retrieved top-k? None if the case has no sections."""
    sections = expected.get("sections") or []
    if not sections:
        return None
    return bool({r["section"] for r in actual["retrieved"]} & set(sections))


def reciprocal_rank(expected: dict, actual: dict) -> float | None:
    """1/rank of the first expected section in the retrieved list (0 if absent).

    hit@3 only says whether the right section was sent to the LLM; this also says how
    high it was ranked, which separates retrieval modes when both reach hit@3 = 1.
    """
    sections = expected.get("sections") or []
    if not sections:
        return None
    for rank, r in enumerate(actual["retrieved"], start=1):
        if r["section"] in sections:
            return 1 / rank
    return 0.0


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation), e.g. p=95."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * p // 100))  # ceil without importing math
    return ordered[int(rank) - 1]
