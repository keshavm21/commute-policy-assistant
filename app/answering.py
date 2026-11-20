"""The question-answering pipeline behind POST /ask.

    retrieve (this company, active version) -> relevance gate -> 1 LLM call
    -> validate output -> (maybe 1 retry) -> citations built in code

Everything that must always hold (which company, which version, how many chunks, what a
citation says, what the user sees on failure) is decided here in code, not by the model.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm import BudgetedLLM, LLMBudgetExceeded, LLMClient, LLMError
from app.prompts import RETRY_SUFFIX, SYSTEM_PROMPT, build_user_prompt, label_chunks
from app.retrieval import Retrieval, RetrievedChunk, Retriever
from app.schemas import AskResponse, Citation

log = logging.getLogger(__name__)

# Fixed user-facing messages: written by us, never by the model.
NOT_COVERED_MSG = (
    "Your company's transport policy does not cover this question. "
    "Please contact your transport desk for help."
)
NO_POLICY_MSG = (
    "There is no active transport policy on file for your company yet. "
    "Please contact your transport desk."
)
TRY_AGAIN_MSG = (
    "Sorry, we couldn't get an answer right now. Please try again in a moment, "
    "or contact your transport desk if it keeps happening."
)
MAX_ANSWER_CHARS = 1200


class MalformedOutput(Exception):
    pass


class _ModelReply(BaseModel):
    status: str
    answer: str = ""
    citations: list[str] = []


@dataclass(frozen=True)
class ValidReply:
    status: str  # "answered" | "not_covered"
    answer: str
    labels: list[str]


def parse_model_reply(raw: str, allowed_labels: set[str], max_citations: int) -> ValidReply:
    """Treat model output as untrusted. Accept it only if every rule holds."""
    text = raw.strip()
    if text.startswith("```"):  # tolerate a ```json fence, nothing more creative
        text = text.strip("`").removeprefix("json").strip()
    try:
        reply = _ModelReply.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise MalformedOutput(f"not the expected JSON: {type(exc).__name__}") from None

    if reply.status == "not_covered":
        return ValidReply("not_covered", "", [])
    if reply.status != "answered":
        raise MalformedOutput(f"unknown status {reply.status!r}")

    answer = reply.answer.strip()
    labels = list(dict.fromkeys(reply.citations))  # de-duplicate, keep order
    if not answer:
        raise MalformedOutput("answered with an empty answer")
    if len(answer) > MAX_ANSWER_CHARS:
        raise MalformedOutput("answer too long")
    if not labels:
        raise MalformedOutput("answered without any citation")
    if len(labels) > max_citations:
        raise MalformedOutput(f"{len(labels)} citations, limit is {max_citations}")
    unknown = [lb for lb in labels if lb not in allowed_labels]
    if unknown:
        raise MalformedOutput(f"cited excerpts that were not provided: {unknown}")
    return ValidReply("answered", answer, labels)


@dataclass
class Trace:
    """Internal diagnostics for logs and the eval. Never sent to the client."""

    request_id: str
    company_id: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    best_distance: float | None = None
    llm_calls: int = 0
    reason: str = ""  # why we ended where we did, e.g. "gate", "llm_timeout"
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class Outcome:
    response: AskResponse
    trace: Trace


class Answerer:
    def __init__(self, retriever: Retriever, llm: LLMClient, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    async def answer(self, company_id: str, question: str) -> Outcome:
        s = self.settings
        start = time.monotonic()
        deadline = start + s.request_deadline_s
        trace = Trace(request_id=uuid.uuid4().hex[:12], company_id=company_id)

        def finish(status: str, answer: str, citations: list[Citation], reason: str) -> Outcome:
            trace.reason = reason
            trace.total_ms = (time.monotonic() - start) * 1000
            log.info("ask request_id=%s company=%s status=%s reason=%s llm_calls=%d ms=%.0f",
                     trace.request_id, company_id, status, reason, trace.llm_calls,
                     trace.total_ms)
            return Outcome(AskResponse(status=status, answer=answer, citations=citations,
                                       request_id=trace.request_id), trace)

        # 1. Retrieval, restricted to the caller's company (from the JWT), with a timeout.
        t0 = time.monotonic()
        try:
            retrieval: Retrieval = await asyncio.wait_for(
                asyncio.to_thread(self.retriever.retrieve, company_id, question, s.max_chunks),
                timeout=min(s.retrieval_timeout_s, deadline - time.monotonic()),
            )
        except TimeoutError:
            return finish("try_again", TRY_AGAIN_MSG, [], "retrieval_timeout")
        except Exception:  # includes IsolationViolation: fail closed, return nothing
            log.exception("retrieval failed request_id=%s", trace.request_id)
            return finish("try_again", TRY_AGAIN_MSG, [], "retrieval_error")
        finally:
            trace.retrieval_ms = (time.monotonic() - t0) * 1000

        if retrieval.policy is None:
            return finish("no_policy", NO_POLICY_MSG, [], "no_active_policy")

        chunks = retrieval.chunks[: s.max_chunks]  # hard cap, whatever the retriever did
        trace.retrieved = chunks
        trace.best_distance = min((c.distance for c in chunks), default=None)

        # 2. Relevance gate in plain code: nothing close enough -> "not covered", 0 LLM calls.
        if trace.best_distance is None or trace.best_distance > s.relevance_max_distance:
            return finish("not_covered", NOT_COVERED_MSG, [], "gate")

        # 3. Ask the LLM; validate; retry at most once if it is worth it and time allows.
        labelled = label_chunks(chunks)
        user_prompt = build_user_prompt(question, labelled)
        llm = BudgetedLLM(self.llm, max_calls=s.max_llm_calls)
        reply: ValidReply | None = None
        reason = ""
        while reply is None:
            remaining = deadline - time.monotonic()
            is_retry = llm.calls > 0
            if remaining <= 0 or (is_retry and remaining < s.min_time_for_retry_s):
                reason = reason or "deadline"
                break
            t1 = time.monotonic()
            try:
                raw = await asyncio.wait_for(
                    asyncio.to_thread(llm.complete, SYSTEM_PROMPT,
                                      user_prompt + (RETRY_SUFFIX if is_retry else ""),
                                      remaining),
                    timeout=remaining,
                )
                reply = parse_model_reply(raw, set(labelled), s.max_chunks)
            except LLMBudgetExceeded:
                break
            except TimeoutError:
                reason = "llm_timeout"
                break  # no time left for a retry anyway
            except LLMError as exc:
                reason = "llm_error"
                log.warning("llm error request_id=%s: %s", trace.request_id, exc)
                if not exc.retryable:
                    break
            except MalformedOutput as exc:
                reason = "malformed_output"
                log.warning("malformed llm output request_id=%s: %s", trace.request_id, exc)
            finally:
                trace.llm_ms += (time.monotonic() - t1) * 1000
                trace.llm_calls = llm.calls

        if reply is None:
            return finish("try_again", TRY_AGAIN_MSG, [], reason or "llm_budget")
        if reply.status == "not_covered":
            return finish("not_covered", NOT_COVERED_MSG, [], "model_not_covered")

        # 4. Citations come from stored metadata, looked up by the labels we handed out.
        citations = [_citation(labelled[label]) for label in reply.labels]
        return finish("answered", reply.answer, citations, "answered")


def _citation(r: RetrievedChunk) -> Citation:
    c = r.chunk
    return Citation(document=c.doc_title, version=c.version, effective_date=c.effective_date,
                    section=c.section, heading=c.heading, excerpt=c.body)
