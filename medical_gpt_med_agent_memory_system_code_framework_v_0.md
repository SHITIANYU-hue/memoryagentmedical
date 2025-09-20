# MedicalGPT + MedAgent Memory System — Code Framework (v0.1)

> End‑to‑end structure for PT → SFT → (DPO/ORPO | RLHF/GRPO) training, with a clinical **Memory/RAG** subsystem (recency × trust × priority), conflict resolution, and agent orchestration.

---

## 0. Repo Layout
```
medicalgpt/
├── README.md
├── pyproject.toml
├── configs/
│   ├── memory.yaml           # ★ 记忆系统权重/冲突/隐私
│   └── serving.yaml
├── src/
│   ├── memory/               # ★ 本次优先完善
│   │   ├── schemas.py        # 记忆体/检索配置/仲裁记录
│   │   ├── privacy.py        # 脱敏/访问控制/审计
│   │   ├── embed/
│   │   │   ├── text_embedder.py
│   │   │   └── image_embedder.py
│   │   ├── store/
│   │   │   ├── vector_store.py   # FAISS/Weaviate 接口
│   │   │   └── graph_store.py    # Neo4j/GraphRAG 接口
│   │   ├── retriever.py      # 混合检索 + 动态加权
│   │   ├── conflict_resolver.py  # 冲突检测/仲裁（肿瘤场景）
│   │   ├── feedback_loop.py  # 个案管理师反馈 → 记忆与权重更新
│   │   └── eval/
│   │       ├── metrics.py    # 命中率/一致性/冲突解决率
│   │       └── runners.py
│   └── serving/
│       └── api/
│           └── memory_api.py # 插入/检索/仲裁/审计 REST
└── tests/
    ├── test_memory_store.py
    ├── test_retriever_scoring.py
    ├── test_conflicts_oncology.py
    └── test_privacy_guard.py
```

---

## 1) Config Examples (YAML)

### `configs/memory.yaml`
```yaml
memory:
  vector_store: weaviate   # or faiss
  graph_store: neo4j
  weights:
    recency: 0.35
    trust:   0.35
    priority:0.20
    modality:0.10
  conflict:
    time_window_days: 30
    value_delta_threshold: 0.20
    arbitration_margin: 0.15
  privacy:
    high:   encrypt:true,  mfa:true
    medium: encrypt:true,  mfa:false
    low:    encrypt:false, mfa:false
```

### `configs/rlhf_grpo.yaml`
```yaml
train:
  base_model: Qwen2.5-Med-7B
  sft_ckpt: checkpoints/sft/epoch-3
  rm_ckpt:  checkpoints/rm/best
  peft: qlora
  bf16: true
  batch_size: 2
  grad_accum: 16
  lr: 1.5e-5
  max_steps: 20000
  clip_grad_norm: 1.0
  rollout:
    max_new_tokens: 512
    temperature: 0.7
  reward:
    coeff_accuracy: 1.0
    coeff_consistency: 0.5
    coeff_safety: 1.5
    penalty_conflict: 1.0
    penalty_privacy: 2.0
  safety:
    med_guideline_gate: true
    allow_drug_dose_output: false # gated by clinician review
```

---

## 2) Key Modules — Code Stubs (Memory Only)

### `src/memory/schemas.py`
```python
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime

PrivacyLevel = Literal["high", "medium", "low"]
SourceType = Literal["EHR", "patient_report", "lab", "radiology"]

@dataclass
class MemoryItem:
    id: str
    patient_id_hash: str
    text: str
    embedding: List[float]
    modalities: List[str]
    source: SourceType
    timestamp: datetime
    recency_days: int
    confidence: float         # 由抽取/来源评估
    source_trust: float       # EHR>lab>report 默认表
    priority: float           # 关键事件↑
    privacy_level: PrivacyLevel
    tags: List[str] = field(default_factory=list)
    original_id: Optional[str] = None
    schema_version: str = "v1.2"

@dataclass
class ArbitrationRecord:
    arb_id: str
    patient_id_hash: str
    key: str                  # 如 tumor_status / CEA
    decided_item_id: Optional[str]
    candidate_item_ids: List[str]
    mode: Literal["auto", "human_required", "human_resolved"]
    arbitrator: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    note: str = ""

@dataclass
class RetrievalConfig:
    w_recency: float = 0.35
    w_trust: float = 0.35
    w_priority: float = 0.20
    w_modality: float = 0.10
    fuse_alpha: float = 0.5    # 向量相似度与元特征评分的融合权重
```

### `src/memory/store/vector_store.py`
```python
from typing import List, Tuple
from ..schemas import MemoryItem

class VectorStore:
    def upsert(self, items: List[MemoryItem]):
        raise NotImplementedError

    def search(self, query: str, top_k: int = 16) -> List[Tuple[MemoryItem, float]]:
        """返回 (item, cosine_sim) 列表"""
        raise NotImplementedError

class FAISSStore(VectorStore):
    def __init__(self, dim: int = 1024):
        # 构建 faiss index, 省略实现
        ...
```

### `src/memory/store/graph_store.py`
```python
from typing import Iterable, Dict, Any, List
from ..schemas import MemoryItem

class GraphStore:
    def upsert_events(self, items: Iterable[MemoryItem]):
        """将记忆体投影为时间线节点: (Patient)-[HAS_EVENT]->(Event {tags,...})"""
        raise NotImplementedError

    def path_timeline(self, patient_id_hash: str) -> List[Dict[str, Any]]:
        """返回时序事件列表(手术、化疗、影像复查、标志物)。"""
        raise NotImplementedError
```

### `src/memory/privacy.py`
```python
from typing import Dict, Any
from ..schemas import MemoryItem

class PrivacyGuard:
    def __init__(self, policy: Dict[str, Any]):
        self.policy = policy  # 来自 configs/memory.yaml

    def redact(self, item: MemoryItem, audience: str) -> MemoryItem:
        # 根据 privacy_level 与 audience (patient/clinician/research) 做字段隐藏
        # high → 加密字段或不返回原文，仅摘要；medium → 局部遮盖；low → 全量
        return item

    def check_output(self, text: str, level: str) -> str:
        # 出口审查：在响应输出前进行PII/药嘱/剂量等拦截与替换
        return text
```

### `src/memory/retriever.py`
```python
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
```

### `src/memory/conflict_resolver.py`（肿瘤特化）
```python
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
```

### `src/memory/feedback_loop.py`
```python
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
```

### `src/serving/api/memory_api.py`
```python
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/memory", tags=["memory"])

class UpsertReq(BaseModel):
    items: List[dict]

class QueryReq(BaseModel):
    query: str
    k: int = 8
    modalities: List[str] = ["text"]

@router.post("/upsert")
def upsert(req: UpsertReq):
    # 校验 + 脱敏 + 入库
    return {"ok": True, "count": len(req.items)}

@router.post("/search")
def search(req: QueryReq):
    # 混合检索 + 返回 evidence 列表（按隐私级别过滤）
    return {"hits": []}

@router.post("/arbitrate")
def arbitrate(payload: dict):
    # 触发/记录仲裁
    return {"status": "queued"}
```

---

## 3) Evaluation (Memory)

- **命中率/召回率**：基于 gold evidence（来自人工标注时间线）。
- **一致性**：回答/证据与患者时间线是否矛盾。
- **冲突解决率**：自动仲裁成功比例、人工介入率、平均决策时延。
- **隐私合规**：不同 audience 的泄露率应为 0（单测 + 集成测试）。

---

## 4) Mini To‑Run 清单

1. 实现 `FAISSStore.search/upsert`（先用句向量 e5‑med/bge 生成 embedding）。
2. 用 `HybridRetriever` 跑通 `/memory/search`，返回前 K 条证据。
3. 写 3 个 `ConflictResolver` 用例：`tumor_status`、`TNM_stage`、`CEA`。
4. 打通 `PrivacyGuard` 出口审查，屏蔽剂量/PII 关键字。
5. 加 `tests/test_conflicts_oncology.py` 的伪造样例，跑 `pytest` 通过。


### `datasets/builders/role_play_data.py`
```python
"""Generate/parse role-play SFT data: doctor–patient dialogs with tags."""
from typing import Dict, Any, Iterable

def build_examples(raw_rows: Iterable[Dict[str, Any]]):
    for r in raw_rows:
        yield {
            "prompt": r["dialog_prompt"],
            "response": r["doctor_answer"],
            "meta": {
                "tumor_type": r["tumor"],
                "stage": r["stage"],
                "tags": r.get("tags", []),
            }
        }
```

### `datasets/builders/rm_pairwise_builder.py`
```python
"""Construct (chosen, rejected) pairs for RM/DPO training from clinician votes."""
```

---

## 4) Training Entrypoints

### `scripts/train_sft.sh`
```bash
python -m src.training.sft.trainer_sft \
  --model Qwen2.5-Med-7B \
  --data data/processed/sft.jsonl \
  --peft qlora --bf16 --batch 4 --grad-accum 8 \
  --save-dir checkpoints/sft
```

### `scripts/train_grpo.sh`
```bash
python -m src.training.rlhf.grpo_loop \
  --policy checkpoints/sft \
  --rm     checkpoints/rm \
  --cfg    configs/rlhf_grpo.yaml
```

---

## 5) Serving (FastAPI)

### `src/serving/api/app.py`
```python
from fastapi import FastAPI
from ..inference.generator import ConstrainedGenerator
from ...agents.med_agent import MedAgent

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(req: dict):
    # construct (retriever, generator, safety) via DI container
    agent: MedAgent = app.state.agent
    out = agent.respond(req["query"], context=req.get("context", {}))
    return out
```

---

## 6) Tests (minimal)

```
pytest -q
```

```python
# tests/test_memory.py
from src.memory.schemas import MemoryItem
from src.memory.retriever import HybridRetriever, RetrievalConfig

def test_meta_score_ordering():
    # fabricate items and assert meta ranking by recency/trust/priority
    assert True
```

---

## 7) Minimal README outline

- Install & setup (poetry/pip, CUDA, Flash‑Attn optional)
- Data preparation (role_play_data, rm_pairwise_builder)
- Train PT → SFT → DPO/ORPO or RLHF/GRPO
- Launch API server and try `/chat`
- Memory system knobs (recency/trust/priority weights, conflict window, privacy gates)
- Safety policy and clinical gating (what is blocked by default)

---

## 8) Next Steps

- Fill reward_fn for GRPO (accuracy/consistency/safety/penalties)
- Implement conflict detection rules for typical oncology variables
- Add Neo4j graph projection for timelines (tumor markers, staging, treatments)
- Wire real embedder models and vector DB client
- Add evaluation runners for clinical tasks and user studies

