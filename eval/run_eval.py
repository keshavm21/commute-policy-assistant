"""Run the evaluation set and write results in the required JSONL format.

    python eval/run_eval.py                      # offline: fake LLM, no API key needed
    python eval/run_eval.py --llm real           # real LLM (needs LLM_API_KEY)
    python eval/run_eval.py --mode vector        # compare retrieval modes

Each run starts from a fresh, temporary store seeded from data/policies, so results do
not depend on whatever state a running server has. Every case goes through the same
path as the API: a JWT is issued and verified, and the company comes from the token.
"""

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.auth import decode_token, issue_token, load_users  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.main import build_services  # noqa: E402
from eval.scoring import (  # noqa: E402
    percentile, reciprocal_rank, retrieval_hit, score_case,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm", choices=["fake", "real"], default="fake")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default=None,
                        help="retrieval mode (default: RETRIEVAL_MODE setting)")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "cases.jsonl")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the whole set N times for latency stats (scores use run 1)")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds between questions, to stay under free-tier rate limits")
    parser.add_argument("--only", nargs="+", default=None, metavar="ID",
                        help="run only these case ids (for quick checks)")
    args = parser.parse_args()

    base = get_settings()
    mode = args.mode or base.retrieval_mode
    out = args.out or ROOT / "eval" / "results" / f"{args.llm}-{mode}.jsonl"
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
        if not cases:
            parser.error("no case matches --only")

    with tempfile.TemporaryDirectory() as tmp:
        settings = base.model_copy(update={
            "var_dir": Path(tmp), "retrieval_mode": mode,
            "llm_provider": "openai_compatible" if args.llm == "real" else "fake",
        })
        svc = build_services(settings)
        users = load_users(settings.users_file)
        employee_for = {p.company_id: p for p in users.values() if p.role == "employee"}
        active_version = {c: svc.registry.get_active(c).version for c in employee_for}

        runs: list[list[dict]] = []  # scored rows for every repeat
        rows, latencies, traces = [], [], []
        for run in range(args.repeat):
            run_rows = []
            for case in cases:
                company, question = case["input"]["company"], case["input"]["question"]
                token = issue_token(employee_for[company], settings)
                principal = decode_token(token, settings)  # same check as the API
                outcome = asyncio.run(svc.answerer.answer(principal.company_id, question))
                latencies.append(outcome.trace.total_ms)
                run_rows.append(_row(case, outcome, company, active_version[company]))
                if run == 0:
                    traces.append(outcome.trace)
                if args.pause:
                    time.sleep(args.pause)
            runs.append(run_rows)
        rows = runs[0]  # the JSONL file holds run 1; the summary covers every run

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summary(rows, traces, latencies, args.llm, mode, svc.answerer.llm.name)
    all_rows = [r for run_rows in runs for r in run_rows]
    summary["all_runs"] = {
        "runs": len(runs),
        "pass_rate_per_run": [round(sum(r["pass"] for r in rr) / len(rr), 3) for rr in runs],
        "failures": dict(Counter(r["id"] for r in all_rows if not r["pass"])),
        "try_again_count": sum(r["actual"]["status"] == "try_again" for r in all_rows),
        "questions": len(all_rows),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _print(summary, rows, out)
    return 0


def _row(case: dict, outcome, company: str, active_version: int) -> dict:
    resp, trace = outcome.response, outcome.trace
    actual = {
        "status": resp.status,
        "answer": resp.answer,
        "cited_sections": [c.section for c in resp.citations],
        "cited_documents": sorted({f"{c.document} v{c.version}" for c in resp.citations}),
        "retrieved": [
            {"company": r.chunk.company_id, "version": r.chunk.version,
             "section": r.chunk.section, "distance": round(r.distance, 3)}
            for r in trace.retrieved
        ],
        "reason": trace.reason,
        "llm_calls": trace.llm_calls,
        "latency_ms": round(trace.total_ms, 1),
    }
    failed = score_case(case["expected"], actual, company, active_version)
    notes = case.get("notes", "")
    if failed:
        notes = f"{notes} | FAILED: {'; '.join(failed)}"
    return {"id": case["id"], "input": case["input"], "expected": case["expected"],
            "actual": actual, "pass": not failed, "notes": notes}


def _summary(rows, traces, latencies, llm, mode, llm_name) -> dict:
    def rate(flags):
        flags = [f for f in flags if f is not None]
        return round(sum(flags) / len(flags), 3) if flags else None

    answerable = [r for r in rows if r["expected"]["status"] == "answered"]
    ranks = [reciprocal_rank(r["expected"], r["actual"]) for r in answerable]
    unanswerable = [r for r in rows if r["expected"]["status"] == "not_covered"]
    all_retrieved = [x for r in rows for x in r["actual"]["retrieved"]]
    return {
        "llm": llm, "llm_client": llm_name, "retrieval_mode": mode, "cases": len(rows),
        "pass_rate": rate([r["pass"] for r in rows]),
        "retrieval": {
            "hit_at_1": rate([None if rr is None else rr == 1.0 for rr in ranks]),
            "hit_at_3": rate([retrieval_hit(r["expected"], r["actual"]) for r in answerable]),
            "mrr": rate(ranks),
            "isolation_rate": rate([x["company"] == r["input"]["company"]
                                    for r in rows for x in r["actual"]["retrieved"]]),
            "chunks_checked": len(all_retrieved),
        },
        "answer": {
            "answerable_status_ok": rate([r["actual"]["status"] == "answered"
                                          for r in answerable]),
            "not_covered_status_ok": rate([r["actual"]["status"] == "not_covered"
                                           for r in unanswerable]),
            "answerable_fully_correct": rate([r["pass"] for r in answerable]),
            "forbidden_content_violations": sum("forbidden:" in r["notes"] for r in rows),
            "isolation_or_freshness_violations": sum(
                "isolation:" in r["notes"] or "freshness:" in r["notes"] for r in rows),
        },
        "budgets": {
            "max_llm_calls": max(t.llm_calls for t in traces),
            "mean_llm_calls": round(sum(t.llm_calls for t in traces) / len(traces), 2),
            "max_cited_chunks": max(len(r["actual"]["cited_sections"]) for r in rows),
            "latency_ms_p50": round(percentile(latencies, 50), 1),
            "latency_ms_p95": round(percentile(latencies, 95), 1),
            "latency_ms_max": round(max(latencies), 1),
            "latency_samples": len(latencies),
        },
        "outcome_reasons": dict(Counter(t.reason for t in traces)),
    }


def _print(summary: dict, rows: list[dict], out: Path) -> None:
    for r in rows:
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"{mark}  {r['id']:<32} {r['actual']['status']:<12} {r['actual']['reason']}")
        if not r["pass"]:
            print(f"      {r['notes'].split('FAILED: ')[-1]}")
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
