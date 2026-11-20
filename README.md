# Commute Policy Assistant

An assistant that answers employee transport questions from **their own company's current
policy only**, cites the document and section, says plainly when the policy is silent, and
fails with a clear "try again" instead of hanging or guessing.

Stack: Python 3.12, FastAPI, Chroma (embedded), all-MiniLM-L6-v2 embeddings (local ONNX),
any OpenAI-compatible LLM (default: `qwen/qwen3.8-27b` on Groq's free tier), PyJWT.

**Measured with the real LLM** (34-case eval set × 3 runs, details in
[eval/REPORT.md](eval/REPORT.md)):
- 99/102 cases passed, with zero chunks from another company or an old version, and zero
  injected or cross-company claims.
- The 3 misses are one case, refused each run: an answer that depends on a detail the
  asker didn't give. It is the known failure documented in the report.
- p95 latency of **696 ms** with the LLM call included (target ≤ 3 s).
- At most 1 LLM call (limit 2) and at most 2 cited chunks (limit 3) per question.

## Run it

```bash
docker compose up
```

That is the only command needed. It builds the image, seeds the three sample policies and
serves the API on http://localhost:8000 (interactive docs at `/docs`).

- **Without an API key**, it starts with an offline *extractive fake LLM*, which answers by
  quoting the top-ranked policy section. Everything else (auth, isolation, versioning,
  citations, failure handling) is the real code.
- **With a real LLM:** `cp .env.example .env`, then put a free Groq key in `LLM_API_KEY`
  (https://console.groq.com/keys) and run `docker compose up` again. `GET /health` shows
  which LLM is in use. Any OpenAI-compatible endpoint works by setting `LLM_BASE_URL` and
  `LLM_MODEL`.

**Tests and evaluation** (no network or API key needed; both use fake LLMs):

```bash
docker compose run --rm api python -m pytest -q          # 152 tests
docker compose run --rm api python eval/run_eval.py      # offline eval -> eval/results/
docker compose run --rm api python eval/run_eval.py --llm real --repeat 3 --pause 5   # needs LLM_API_KEY
```

To run a single test, use for example `python -m pytest tests/test_api.py -k company -q`.
Local development without Docker:
`uv venv --python 3.12 .venv && uv pip install -r requirements.txt`, then
`.venv/bin/uvicorn app.main:app --reload`.

### Try it

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/token -H 'content-type: application/json' \
  -d '{"employee_id":"M2001"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

curl -s -X POST localhost:8000/ask -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"question":"Can I book a cab on Saturday?"}'
```

Demo users (`data/users.json`): `Z1001` / `M2001` / `K3001` are employees of Zentrix,
Meridian and Kestrel. `Z9001` / `M9001` / `K9001` are their admins, who can upload a new
policy version with `POST /admin/policies {"content": "<markdown>"}`.

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /auth/token` | none | Mock identity provider: employee id → signed JWT |
| `POST /ask` | employee or admin | `{"question": "..."}` → answer + citations |
| `POST /admin/policies` | admin | Ingest a new version of the admin's own company policy |
| `GET /admin/policies` | admin | Version history for the admin's company |
| `GET /health` | none | Liveness, plus which LLM and retrieval mode are active |

`/ask` always returns the same shape:

```json
{"status": "answered | not_covered | no_policy | try_again",
 "answer": "...",
 "citations": [{"document": "...", "version": 2, "effective_date": "...",
                "section": "4.2", "heading": "...", "excerpt": "<the policy text itself>"}],
 "request_id": "..."}
```

The status codes are 200 (answered or not covered), 503 (try again), 401 (bad or missing
token) and 422 (bad input, including any attempt to send `company_id`).

## Design

```
JWT ─► company_id (server-trusted, never from the request)
question ─► registry: active collection for company_id  ─(none)─► no_policy
         ─► vector search in THAT collection only, top 3
         ─► check every chunk's company + version            ─(mismatch)─► try_again
         ─► relevance gate: best distance > 0.65?            ─► not_covered (0 LLM calls)
         ─► LLM call #1 → JSON {status, answer, citations: ["C1".."C3"]}
         ─► validate output ─(invalid, time left)─► LLM call #2 (retry)
         ─► citations built in code from stored chunk metadata
```

**The rule behind every decision:** anything that must always be true is enforced in
plain code. The LLM does only the job code cannot do: it reads up to three sections,
judges whether they answer the question, and phrases the answer.

### 1. Company isolation is structural, not a filter
Each (company, version) pair has **its own Chroma collection**. `/ask` reads the
company from the verified JWT, looks up that company's active collection in the
registry, and searches only that collection. Another company's chunks are not filtered
out; they live somewhere this request never opens. As a second safeguard, every
returned chunk's `company_id` and `version` metadata is checked, and a mismatch fails
the request closed (`try_again`, nothing returned). The request model uses
`extra="forbid"`, so a body that contains `company_id` is rejected with a 422 instead of
being silently ignored. Query strings and headers are never read.

### 2. Versioning and atomic ingestion
A SQLite table (`app/registry.py`) records every version with status
`building | active | retired | failed`. A partial unique index makes "one active version
per company" a database guarantee. Ingestion (`app/ingestion.py`) runs in this order:

1. Parse and validate with no side effects: front matter, a newer version number, and
   non-empty, unique numbered sections. A bad document is rejected and nothing changes.
2. Insert a `building` row.
3. Build a **new** collection and verify the stored chunk count.
4. Switch the active version (retire the old, activate the new) in **one SQLite
   transaction**.

If steps 2 to 3 fail, the new collection is deleted and the row is marked `failed`. The
old version was never touched, so it keeps answering. If the process crashes mid-ingest,
the leftover `building` rows are cleaned up at startup. Re-uploading an identical file is
a no-op, detected by its SHA-256 hash. An older or equal version number is rejected.

### 3. Grounded answers with checkable citations
Policies are Markdown with numbered sections (`## 4.2 Security escort ...`), and **one
section is one chunk**, so a citation is exact by construction. The LLM sees the
excerpts labelled `C1`–`C3` and cites labels only. Code maps the labels back to document,
version and section, and returns the section text itself as `excerpt`, so the employee
can check the answer. A label that was never handed out fails validation, so the model
cannot invent a citation.

### 4. "Not covered" in two layers
- **Code:** if the best chunk's cosine distance is above 0.65, the reply is `not_covered`
  without calling the LLM.
- **LLM:** for related but non-answering sections, the model returns
  `status: "not_covered"`.

Both produce the same fixed message pointing to the transport desk. The gate is
deliberately loose; see the trade-offs.

### 5. Untrusted model output and prompt injection
- The model's reply must be JSON with an allowed status, an answer that is non-empty and
  within a length limit, and 1 to 3 citations from the labels provided. Anything else is
  treated as malformed.
- **Quarantine at ingestion (code).** A section whose text addresses an AI ("ignore
  previous instructions", "AI assistant", "system notice"…) is not indexed. It is logged
  and reported to the admin in `quarantined_sections`. Kestrel's policy contains such a
  planted section (§9.2), and it never reaches the model.
- **Prompt (second layer).** Excerpts and the question are wrapped in tags. The system
  prompt says they are data, and that a rule addressed to an AI is not a policy rule.
  Tag-like text is stripped from documents and questions, so they cannot close a tag
  early.
- **Why both layers.** In the first real-LLM run, the model *did* obey the planted
  injection ("Yes, all cab trips are free and unlimited"). The fix and its measurement are
  in the report.
- **Structural guarantee.** Even a fully obeyed injection cannot reach another company's
  text, because that text is never in the prompt.

### 6. Failures and the two budgets
- **At most 2 LLM calls per question.** The normal path makes 1 call, or 0 when the gate
  answers. The only second call is a single retry after malformed output or a transient
  5xx/network error, and only if at least 1 s of the deadline remains. A 429 (quota) error
  is not retried. A per-question `BudgetedLLM` wrapper raises on a third call, and a test
  checks it.
- **At most 3 cited chunks.** Retrieval returns 3 chunks, the pipeline slices to 3 again,
  and validation rejects more than 3 citations.
- **Within 3 s at p95.** The whole pipeline has a 2.7 s deadline. Retrieval runs under its
  own timeout, and the LLM call gets the remaining time as its HTTP timeout. When time
  runs out, the user gets `try_again` (HTTP 503). Responses are never streamed, so a half
  answer is impossible. The embedding model loads at startup, so the first question is
  not slow. Measured latencies are in [eval/REPORT.md](eval/REPORT.md).

## Evaluation

- [eval/cases.jsonl](eval/cases.jsonl) has 34 hand-written cases: direct questions,
  paraphrases, cross-company traps, version checks, "not covered" questions, document
  and user injections, and conditional answers.
- `eval/run_eval.py` sends every case through the same code path as the API (a JWT is
  issued and verified for each case) on a freshly seeded store.
- It writes `eval/results/<llm>-<mode>.jsonl` with the required fields `id, input,
  expected, actual, pass, notes`, plus a `.summary.json` with the metrics.
- **The submitted evaluation set with real-LLM results is
  [eval/results/real-vector-qwen3.8-27b.jsonl](eval/results/real-vector-qwen3.8-27b.jsonl).**
  The other files in `eval/results/` are the model comparison and the offline
  vector/hybrid runs.
- Scoring is deterministic (no LLM judge). Method, results, known failures and the
  vector-vs-hybrid comparison are in **[eval/REPORT.md](eval/REPORT.md)**.

## Trade-offs

- **A collection per version, instead of one collection with metadata filters.** Isolation
  and "only the active version answers" become structural and easy to verify. The cost is
  more collections, and retired versions are kept on disk. Keeping them is deliberate: a
  request that read the old pointer can still finish, and the history is auditable.
- **A deliberately loose relevance gate (0.65).** On the eval set, some unanswerable
  questions sit *closer* to a policy section than the hardest answerable one (0.38 against
  0.56), so no threshold separates the two groups. A wrongly gated answerable question can
  never recover, so the gate catches only clearly off-topic questions and leaves the
  judgement to the LLM.
- **Deterministic string-match scoring instead of an LLM judge.** It is free,
  reproducible and has no quota. The cost is that a correct answer worded unexpectedly can
  fail; each case lists several acceptable phrasings to reduce this.
- **A hybrid search bonus that was measured and switched off.** Hybrid search lowered
  hit@1 from 1.00 to 0.875 (MRR from 1.00 to 0.92) on this set, with no latency
  difference beyond noise, so `vector` is the default (`RETRIEVAL_MODE=hybrid` turns it
  on). Details are in the report.
- **The LLM was chosen by measurement.** qwen3.8-27b and gpt-oss-120b are about equally
  accurate on the set: 99 against 97 of 102, and neither ever gave a wrong company's rule.
  qwen's p95 was 0.70 s with no deadline timeouts, against 1.22 s and 1 timeout for
  gpt-oss-120b, so qwen is the default.
- **Refusing rather than guessing.** When an answer depends on something the employee
  didn't say, the model sometimes refuses. That sends them to the transport desk, which
  is unhelpful but never wrong. Prompting further to fix two known cases would overfit
  them, so it is reported as a known failure instead.
- **Embedded Chroma in one process**, with no separate vector DB container. That means
  one container and no network hop. The cost is a single process, which this scale does
  not need beyond.
- **A mock identity provider.** `/auth/token` issues a token for any known employee id with
  no password, because the brief asks for a mock issuer. In production, identity would
  come from SSO. Only token *verification* here is production-shaped: the algorithm is
  pinned to HS256, and `exp`, `iss` and `aud` are required.

## Layout

```
app/        main.py (routes, startup) · answering.py (the /ask pipeline) · retrieval.py
            ingestion.py · registry.py (SQLite versions) · store.py (Chroma) · llm.py
            prompts.py · auth.py · schemas.py · config.py (all settings, env-driven)
data/       policies/*.md (3 companies, Zentrix has v1 + v2) · users.json
eval/       cases.jsonl · run_eval.py · scoring.py · results/ · REPORT.md
tests/      152 tests, all offline with fake LLMs
scripts/    issue_token.py
```
