"""Hybrid retrieval: BM25 + dense (FAISS) -> Reciprocal Rank Fusion -> cross-encoder re-ranking."""
import pickle
import re
from collections import defaultdict

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import CFG

_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "is", "are", "was", "were", "be", "that", "this", "with", "as", "at", "it"}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOP]


class HybridRetriever:
    MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

    def __init__(self, cfg=CFG):
        self.cfg = cfg
        emb = HuggingFaceEmbeddings(model_name=cfg.embedding_model, encode_kwargs={"normalize_embeddings": True})
        self.vs = FAISS.load_local(str(cfg.index_dir), emb, allow_dangerous_deserialization=True)
        with open(cfg.index_dir / "chunks.pkl", "rb") as f:
            self.chunks: list[Document] = pickle.load(f)
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])
        self.reranker = CrossEncoder(cfg.reranker_model)

    # ---- individual retrievers (return ordered chunk ids) ----
    def _dense_ids(self, query: str) -> list[int]:
        docs = self.vs.similarity_search(query, k=self.cfg.top_k_dense)
        return [d.metadata["chunk_id"] for d in docs]

    def _bm25_ids(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][: self.cfg.top_k_bm25]
        return [int(i) for i in top if scores[i] > 0]

    # ---- fusion ----
    def _rrf(self, *rankings: list[int]) -> list[int]:
        fused: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] += 1.0 / (self.cfg.rrf_k + rank + 1)
        return [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    # ---- public API ----
    def retrieve(self, query: str, mode: str = "hybrid_rerank", k: int | None = None) -> list[Document]:
        k = k or self.cfg.final_k
        assert mode in self.MODES, f"mode must be one of {self.MODES}"

        if mode == "dense":
            ids = self._dense_ids(query)[:k]
        elif mode == "bm25":
            ids = self._bm25_ids(query)[:k]
        else:
            fused = self._rrf(self._dense_ids(query), self._bm25_ids(query))
            if mode == "hybrid":
                ids = fused[:k]
            else:
                cand = fused[: self.cfg.rerank_candidates]
                if not cand:
                    return []
                scores = self.reranker.predict([(query, self.chunks[i].page_content) for i in cand])
                order = np.argsort(scores)[::-1][:k]
                out = []
                for j in order:
                    c = self.chunks[cand[j]]
                    out.append(Document(page_content=c.page_content, metadata={**c.metadata, "score": float(scores[j])}))
                return out

        return [Document(page_content=self.chunks[i].page_content, metadata={**self.chunks[i].metadata, "score": None}) for i in ids]
