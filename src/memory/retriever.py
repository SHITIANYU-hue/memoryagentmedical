import math
from typing import List, Tuple
from .schemas import MemoryItem, RetrievalConfig

class HybridRetriever:
    def __init__(self, vector_store, bm25_index, cfg: RetrievalConfig):
        self.vs = vector_store
        self.bm25 = bm25_index
        self.cfg = cfg

    def _meta_score(self, m: MemoryItem, query_modalities: List[str]) -> float:
        recency = math.exp(-m.recency_days / 90.0)
        trust   = m.source_trust
        pri     = m.priority
        mod     = 1.0 if any(md in m.modalities for md in query_modalities) else 0.7
        return (self.cfg.w_recency*recency + self.cfg.w_trust*trust +
                self.cfg.w_priority*pri + self.cfg.w_modality*mod)

    def retrieve(self, query: str, k: int = 8, query_modalities: List[str] = ["text"]):
        v_hits: List[Tuple[MemoryItem, float]] = self.vs.search(query, top_k=32)
        b_hits: List[Tuple[MemoryItem, float]] = self.bm25.search(query, top_k=32)
        merged = {}
        for it, sim in v_hits + b_hits:
            meta = self._meta_score(it, query_modalities)
            score = self.cfg.fuse_alpha*sim + (1-self.cfg.fuse_alpha)*meta
            if it.id not in merged or score > merged[it.id][1]:
                merged[it.id] = (it, score)
        ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return ranked[:k]