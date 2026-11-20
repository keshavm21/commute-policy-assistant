"""Thin wrapper around an embedded Chroma client.

Every (company, version) pair gets its own collection. Retrieval only ever opens the
collection the registry marks as active for the caller's company, so another
company's chunks are not filtered out: they live somewhere the code never opens.
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: a policy section (or part of a long one)."""

    chunk_id: str  # unique within a collection, e.g. "4.2" or "4.2#p2"
    text: str  # what gets embedded and shown to the LLM (heading + body)
    company_id: str
    version: int
    doc_title: str
    effective_date: str
    section: str  # e.g. "4.2"
    heading: str  # e.g. "Security escort for women employees"
    body: str  # section text without the heading, used for the user-facing excerpt

    def metadata(self) -> dict:
        return {
            "company_id": self.company_id, "version": self.version, "doc_title": self.doc_title,
            "effective_date": self.effective_date, "section": self.section,
            "heading": self.heading, "body": self.body,
        }

    @classmethod
    def from_chroma(cls, chunk_id: str, text: str, meta: dict) -> "Chunk":
        return cls(
            chunk_id=chunk_id, text=text, company_id=meta["company_id"], version=meta["version"],
            doc_title=meta["doc_title"], effective_date=meta["effective_date"],
            section=meta["section"], heading=meta["heading"], body=meta["body"],
        )


class VectorStore:
    def __init__(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # Local ONNX all-MiniLM-L6-v2: free, offline, no PyTorch. Same model for docs and queries.
        self._embed = DefaultEmbeddingFunction()

    def warm_up(self) -> None:
        """Load the embedding model now so the first real question is not slow."""
        self._embed(["warm up"])

    def create_collection(self, name: str, chunks: list[Chunk]) -> int:
        """Create a brand-new collection holding `chunks`; returns the stored count."""
        col = self._client.create_collection(
            name=name,
            embedding_function=self._embed,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"company_id": chunks[0].company_id, "version": chunks[0].version},
        )
        col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )
        return col.count()

    def delete_collection(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception:  # already gone (or never created) is fine during cleanup
            pass

    def query(self, name: str, text: str, n_results: int) -> list[tuple[Chunk, float]]:
        """Nearest chunks in one collection, as (chunk, cosine distance), closest first."""
        col = self._client.get_collection(name=name, embedding_function=self._embed)
        n = min(n_results, col.count())
        if n == 0:
            return []
        res = col.query(
            query_texts=[text], n_results=n, include=["documents", "metadatas", "distances"]
        )
        return [
            (Chunk.from_chroma(cid, doc, meta), dist)
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def all_chunks(self, name: str) -> list[Chunk]:
        col = self._client.get_collection(name=name, embedding_function=self._embed)
        res = col.get(include=["documents", "metadatas"])
        return [
            Chunk.from_chroma(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
        ]

    def collection_names(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]
