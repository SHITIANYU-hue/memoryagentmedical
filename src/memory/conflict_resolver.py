from dataclasses import dataclass
from typing import List, Dict, Tuple
from .schemas import MemoryItem

# 典型冲突键: 肿瘤状态/分期/标志物CEA/CA19-9/基因变异/治疗方案
CONFLICT_KEYS = {
    "tumor_status": {"exclusive": ["progression","stable","partial_response","complete_response"]},
    "TNM_stage":    {"order": ["I","II","III","IV"]},
    "CEA":          {"numeric": True, "delta_thresh": 0.2},   # 相对变化阈值 20%
}

@dataclass
class Conflict:
    key: str
    items: List[MemoryItem]

class ConflictResolver:
    def __init__(self, time_window_days: int, arbitration_margin: float):
        self.win = time_window_days
        self.margin = arbitration_margin

    def group_by_key(self, items: List[MemoryItem]) -> Dict[str, List[MemoryItem]]:
        buckets: Dict[str, List[MemoryItem]] = {}
        for it in items:
            for t in it.tags:
                k = t.split(":")[0]
                if k in CONFLICT_KEYS:
                    buckets.setdefault(k, []).append(it)
        return buckets

    def detect(self, items: List[MemoryItem]) -> List[Conflict]:
        conflicts: List[Conflict] = []
        grouped = self.group_by_key(items)
        for k, arr in grouped.items():
            spec = CONFLICT_KEYS[k]
            if "exclusive" in spec:
                labels = set([t for it in arr for t in it.tags if t.startswith(k+":")])
                values = {l.split(":")[1] for l in labels}
                if len(values) > 1:
                    conflicts.append(Conflict(k, arr))
            if spec.get("numeric"):
                # 同窗内数值差异>阈值
                arr_sorted = sorted(arr, key=lambda x: x.timestamp)
                vals = [self._extract_numeric(k, it.tags) for it in arr_sorted]
                vmin, vmax = min(vals), max(vals)
                if vmin and vmax and (vmax - vmin)/max(vmin, 1e-6) > spec["delta_thresh"]:
                    conflicts.append(Conflict(k, arr))
        return conflicts

    def arbitrate(self, conflict: Conflict) -> Dict:
        ranked = sorted(conflict.items, key=lambda m: m.confidence * (1.0/(1+m.recency_days)), reverse=True)
        if len(ranked) < 2:
            return {"decision": ranked[0] if ranked else None, "mode": "auto"}
        top, sec = ranked[0], ranked[1]
        gap = (top.confidence/(1+top.recency_days)) - (sec.confidence/(1+sec.recency_days))
        if gap < self.margin:
            return {"decision": None, "mode": "human_required", "candidates": [top, sec]}
        return {"decision": top, "mode": "auto"}

    def _extract_numeric(self, key: str, tags: List[str]):
        for t in tags:
            if t.startswith(key+":"):
                try:
                    return float(t.split(":")[1])
                except: pass
        return None