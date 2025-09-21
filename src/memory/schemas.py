from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime

PrivacyLevel = Literal["high", "medium", "low"]
SourceType = Literal["EHR", "patient_report", "lab", "radiology"]

@dataclass
class MemoryItem:
    id: str # 记忆项唯一ID
    patient_id_hash: str # 患者ID哈希（不是明文，保护隐私）
    text: str # 记忆内容（如“2024-04-01 CEA: 5.2 ng/mL”）
    embedding: List[float] # 文本的向量表示（用于语义检索）
    modalities: List[str] # 模态（如“text”文本、“image”影像报告）
    source: SourceType # 来源（EHR电子病历＞lab化验＞patient_report患者自述）
    timestamp: datetime # 生成时间
    recency_days: int # 距现在的天数（计算新鲜度）
    confidence: float         # 由抽取/来源评估 # 内容可信度（如CT报告可信度0.9，患者自述0.6）
    source_trust: float       # EHR>lab>report 默认表 # 来源可信度（EHR默认1.0，患者自述0.7）
    priority: float           # 关键事件↑ # 优先级（手术/化疗等关键事件1.0，常规随访0.3）
    privacy_level: PrivacyLevel # 隐私级别（high/medium/low）
    tags: List[str] = field(default_factory=list) # 标签（用于冲突检测，如“CEA:5.2”“tumor_status:stable”）
    original_id: Optional[str] = None
    schema_version: str = "v1.2"

@dataclass
class ArbitrationRecord: # 冲突仲裁记录
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
class RetrievalConfig: # 检索时的权重配置
    w_recency: float = 0.35
    w_trust: float = 0.35
    w_priority: float = 0.20
    w_modality: float = 0.10
    fuse_alpha: float = 0.5    # 向量相似度与元特征评分的融合权重