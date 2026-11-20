"""Company-scoped retrieval.

The caller passes the company_id from the verified JWT. We look up THAT company's
active collection in the registry and search only it. After searching we still check
every chunk's metadata (defence in depth): if anything from another company or another
version ever came back, we raise instead of answering.

Two modes, switchable by config so they can be compared on the same eval set:
  vector - cosine similarity on MiniLM embeddings
  hybrid - vector ranking + BM25 keyword ranking, merged with reciprocal rank fusion
"""

import re
import threading
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.registry import PolicyVersion, Registry
from app.store import Chunk, VectorStore

RRF_K = 60  # standard constant from the RRF paper; damps the effect of top ranks
_STOPWORDS = frozenset(
    "a an the is are am i my me can do does what who how when if of to for on in at by "
    "and or be it this that with from will should would".split()
)


class IsolationViolation(Exception):
    """A chunk from the wrong company or version came back. Must never happen."""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    distance: float  # cosine distance from the question (lower = more similar)


@dataclass(frozen=True)
class Retrieval:
    policy: PolicyVersion | None  # None: this company has no active policy
    chunks: list[RetrievedChunk]  # best first, at most k


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class Retriever:
    def __init__(self, registry: Registry, store: VectorStore, mode: str = "vector"):
        if mode not in ("vector", "hybrid"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        self.registry = registry
        self.store = store
        self.mode = mode
        # A collection never changes after it is activated, so its BM25 index can be
        # cached forever under the collection name. A new version = a new name.
        self._bm25_cache: dict[str, tuple[BM25Okapi, list[Chunk]]] = {}
        self._cache_lock = threading.Lock()

    def retrieve(self, company_id: str, question: str, k: int) -> Retrieval:
        policy = self.registry.get_active(company_id)
        if policy is None:
            return Retrieval(policy=None, chunks=[])

        if self.mode == "vector":
            hits = self.store.query(policy.collection_name, question, n_results=k)
            ranked = [RetrievedChunk(chunk, dist) for chunk, dist in hits]
        else:
            ranked = self._hybrid(policy, question, k)

        for r in ranked:  # defence in depth: verify, never assume
            if r.chunk.company_id != company_id or r.chunk.version != policy.version:
                raise IsolationViolation(
                    f"got {r.chunk.company_id} v{r.chunk.version} while serving "
                    f"{company_id} v{policy.version}"
                )
        return Retrieval(policy=policy, chunks=ranked)

    def _hybrid(self, policy: PolicyVersion, question: str, k: int) -> list[RetrievedChunk]:
        bm25, chunks = self._bm25_index(policy.collection_name)

        # Vector ranking over the whole (small) collection, so every chunk has a distance.
        vector_hits = self.store.query(policy.collection_name, question, n_results=len(chunks))
        distance = {c.chunk_id: d for c, d in vector_hits}
        vector_rank = {c.chunk_id: rank for rank, (c, _) in enumerate(vector_hits)}

        # Keyword ranking. Chunks with no shared words get no BM25 rank at all.
        scores = bm25.get_scores(tokenize(question))
        keyword_order = sorted(
            (i for i in range(len(chunks)) if scores[i] > 0), key=lambda i: -scores[i]
        )
        keyword_rank = {chunks[i].chunk_id: rank for rank, i in enumerate(keyword_order)}

        def rrf(chunk_id: str) -> float:
            score = 1 / (RRF_K + vector_rank[chunk_id] + 1)
            if chunk_id in keyword_rank:
                score += 1 / (RRF_K + keyword_rank[chunk_id] + 1)
            return score

        by_id = {c.chunk_id: c for c in chunks}
        best = sorted(by_id, key=lambda cid: (-rrf(cid), distance[cid]))[:k]
        return [RetrievedChunk(by_id[cid], distance[cid]) for cid in best]

    def _bm25_index(self, collection_name: str) -> tuple[BM25Okapi, list[Chunk]]:
        with self._cache_lock:
            if collection_name not in self._bm25_cache:
                chunks = self.store.all_chunks(collection_name)
                bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
                self._bm25_cache[collection_name] = (bm25, chunks)
            return self._bm25_cache[collection_name]
