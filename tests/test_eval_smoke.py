"""Runs the real eval runner offline (fake LLM) and checks the hard guarantees.

Answer quality with the fake LLM is not asserted here (the fake cannot judge "not
covered"); isolation, freshness, budgets and the output format are.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FIELDS = {"id", "input", "expected", "actual", "pass", "notes"}


def test_offline_eval_runs_and_keeps_the_hard_guarantees(tmp_path):
    out = tmp_path / "results.jsonl"
    proc = subprocess.run(
        [sys.executable, "eval/run_eval.py", "--llm", "fake", "--mode", "vector",
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    cases = (ROOT / "eval" / "cases.jsonl").read_text().splitlines()
    assert len(rows) == len(cases) >= 10
    assert all(set(r) == REQUIRED_FIELDS for r in rows)
    assert all(isinstance(r["pass"], bool) for r in rows)

    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert summary["retrieval"]["isolation_rate"] == 1.0
    assert summary["answer"]["isolation_or_freshness_violations"] == 0
    assert summary["budgets"]["max_llm_calls"] <= 2
    assert summary["budgets"]["max_cited_chunks"] <= 3
