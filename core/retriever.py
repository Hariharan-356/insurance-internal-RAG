"""Hybrid retrieval: Chroma vector search + BM25, fused with Reciprocal Rank Fusion.

ACCESS CONTROL happens here, *before* anything can reach the LLM:
  * Chroma query is filtered with access_rank <= user_rank
  * BM25 index is built only from chunks the user is allowed to read
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import settings

_STOP = set(
    "a an the of to in on for and or is are was were be been it this that with as by at "
    "from what which who how do does can i my we our you your if".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass
class Hit:
    doc: Document
    rrf_score: float
    vector_sim: Optional[float] = None
    bm25_score: Optional[float] = None


class HybridRetriever:
    def __init__(self, vector_store):
        self.store = vector_store
        self._bm25_cache: Dict[int, tuple] = {}

    def reset_cache(self):
        self._bm25_cache.clear()

    # ---- BM25 -------------------------------------------------------------
    def _bm25_index(self, max_rank: int):
        if max_rank not in self._bm25_cache:
            try:
                data = self.store.get(
                    where={"access_rank": {"$lte": max_rank}},
                    include=["documents", "metadatas"],
                )
            except Exception:
                data = {"documents": [], "metadatas": []}
            raw_docs = data.get("documents") or []
            raw_metas = data.get("metadatas") or []
            docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(raw_docs, raw_metas)
            ]
            tokenized_corpus = [tokenize(d.page_content) for d in docs]
            bm25 = BM25Okapi(tokenized_corpus) if docs and any(tokenized_corpus) else None
            self._bm25_cache[max_rank] = (bm25, docs)
        return self._bm25_cache[max_rank]

    def _bm25_search(self, query: str, max_rank: int):
        bm25, docs = self._bm25_index(max_rank)
        tokens = tokenize(query)
        if bm25 is None or not tokens or not docs:
            return []
        scores = bm25.get_scores(tokens)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        valid = [
            (d, float(s)) for d, s in ranked
            if float(s) >= settings.bm25_min_score
        ]
        return valid[: settings.bm25_k]

    # ---- Vector -----------------------------------------------------------
    def _vector_search(self, query: str, max_rank: int):
        results = self.store.similarity_search_with_score(
            query, k=settings.vector_k, filter={"access_rank": {"$lte": max_rank}}
        )
        out = []
        for doc, dist in results:
            sim = 1.0 - float(dist)
            if sim >= settings.vector_min_sim:
                out.append((doc, sim))
        return out

    # ---- Fusion -----------------------------------------------------------
    def retrieve(self, query: str, max_rank: int) -> List[Hit]:
        vec = self._vector_search(query, max_rank)
        bm = self._bm25_search(query, max_rank)

        fused: Dict[str, Hit] = {}
        for results, field in ((vec, "vector_sim"), (bm, "bm25_score")):
            for rank, (doc, score) in enumerate(results, start=1):
                cid = doc.metadata.get("chunk_id", str(id(doc)))
                hit = fused.setdefault(cid, Hit(doc=doc, rrf_score=0.0))
                hit.rrf_score += 1.0 / (settings.rrf_k + rank)
                setattr(hit, field, score)

        query_tokens = set(tokenize(query))
        for hit in fused.values():
            doc_tokens = set(tokenize(hit.doc.page_content))
            hit.rrf_score += 0.25 * len(query_tokens & doc_tokens)

        hits = sorted(fused.values(), key=lambda h: h.rrf_score, reverse=True)
        hits = hits[: settings.final_k]
        # Defensive re-check: never return a chunk above the user's clearance.
        return [h for h in hits if int(h.doc.metadata.get("access_rank", 99)) <= max_rank]
