from typing import Dict, Any
from .schemas import ArbitrationRecord, MemoryItem

class FeedbackLoop:
    def __init__(self, vector_store, weight_state: Dict[str, float]):
        self.vs = vector_store
        self.w = weight_state  # {w_recency, w_trust, ...}

    def log_arbitration(self, record: ArbitrationRecord):
        # 记录仲裁与审计
        pass

    def learn_weights(self, signals: Dict[str, float]):
        # RLHF/PBT: 根据命中率、一致性、仲裁率等渐进更新 w_*（带上下限）
        for k, delta in signals.items():
            if k in self.w:
                self.w[k] = float(min(max(self.w[k] + delta, 0.05), 0.85))
        return self.w