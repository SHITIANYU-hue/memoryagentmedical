from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.memory.schemas import MemoryItem, ArbitrationRecord, RetrievalConfig
from src.memory.store.vector_store import VectorStore
from src.memory.retriever import HybridRetriever
from src.memory.conflict_resolver import ConflictResolver, Conflict
from src.memory.feedback_loop import FeedbackLoop
from src.memory.privacy import PrivacyGuard
from uuid import uuid4
import json

router = APIRouter(prefix="/memory", tags=["memory"])


# 依赖项 - 实际应用中应使用依赖注入容器
def get_vector_store() -> VectorStore:
    """获取向量存储实例（实际应用中需替换为真实实现）"""
    # 此处为示例，实际应从应用状态获取已初始化的实例
    from src.memory.store.vector_store import FAISSStore
    return FAISSStore(dim=1024)


def get_retriever(vs: VectorStore = Depends(get_vector_store)) -> HybridRetriever:
    """获取混合检索器实例"""
    config = RetrievalConfig()  # 实际应用中应从配置加载
    # 此处简化BM25索引的初始化
    return HybridRetriever(vector_store=vs, bm25_index=None, cfg=config)


def get_conflict_resolver() -> ConflictResolver:
    """获取冲突解决器实例"""
    # 从配置加载参数（示例值）
    return ConflictResolver(time_window_days=30, arbitration_margin=0.15)


def get_feedback_loop(vs: VectorStore = Depends(get_vector_store)) -> FeedbackLoop:
    """获取反馈循环实例"""
    initial_weights = {
        "w_recency": 0.35,
        "w_trust": 0.35,
        "w_priority": 0.20,
        "w_modality": 0.10
    }
    return FeedbackLoop(vector_store=vs, initial_weights=initial_weights)


def get_privacy_guard() -> PrivacyGuard:
    """获取隐私保护实例"""
    policy = {
        "high": {"encrypt": True, "mfa": True},
        "medium": {"encrypt": True, "mfa": False},
        "low": {"encrypt": False, "mfa": False}
    }
    return PrivacyGuard(policy=policy)


# 请求/响应模型
class MemoryItemCreate(BaseModel):
    patient_id_hash: str
    text: str
    modalities: List[str]
    source: str = Field(..., pattern="^(EHR|patient_report|lab|radiology)$")
    timestamp: datetime
    recency_days: int
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_trust: float = Field(..., ge=0.0, le=1.0)
    priority: float = Field(..., ge=0.0, le=1.0)
    privacy_level: str = Field(..., pattern="^(high|medium|low)$")
    tags: List[str] = []
    original_id: Optional[str] = None


class MemoryItemResponse(BaseModel):
    id: str
    patient_id_hash: str
    text: str
    modalities: List[str]
    source: str
    timestamp: datetime
    recency_days: int
    confidence: float
    source_trust: float
    priority: float
    privacy_level: str
    tags: List[str]
    original_id: Optional[str]

    class Config:
        from_attributes = True


class UpsertResponse(BaseModel):
    ok: bool
    count: int
    item_ids: List[str]


class QueryRequest(BaseModel):
    query: str
    patient_id_hash: Optional[str] = None  # 可选，限定患者
    k: int = Field(8, ge=1, le=100)
    modalities: List[str] = ["text"]
    include_private: bool = False  # 是否包含高隐私级别内容（需权限验证）


class QueryResponse(BaseModel):
    hits: List[Dict[str, Any]]  # 包含记忆项和相关性分数
    retrieval_config: RetrievalConfig


class ArbitrationRequest(BaseModel):
    patient_id_hash: str
    key: str
    decided_item_id: Optional[str]
    candidate_item_ids: List[str]
    note: str = ""
    arbitrator: str  # 仲裁者ID/标识


class ArbitrationResponse(BaseModel):
    arb_id: str
    status: str
    timestamp: datetime


class ConfidenceUpdateRequest(BaseModel):
    item_id: str
    delta: float = Field(..., ge=-1.0, le=1.0)


# API端点
@router.post("/upsert", response_model=UpsertResponse)
def upsert_memory(
        items: List[MemoryItemCreate],
        vector_store: VectorStore = Depends(get_vector_store),
        privacy_guard: PrivacyGuard = Depends(get_privacy_guard)
):
    """插入或更新记忆项（自动处理隐私脱敏）"""
    try:
        memory_items = []
        item_ids = []

        for item_data in items:
            # 生成唯一ID（如果没有）
            item_id = str(uuid4())
            item_ids.append(item_id)

            # 创建MemoryItem对象
            memory_item = MemoryItem(
                id=item_id,
                patient_id_hash=item_data.patient_id_hash,
                text=item_data.text,
                embedding=[],  # 实际应用中应在此处生成嵌入向量
                modalities=item_data.modalities,
                source=item_data.source,
                timestamp=item_data.timestamp,
                recency_days=item_data.recency_days,
                confidence=item_data.confidence,
                source_trust=item_data.source_trust,
                priority=item_data.priority,
                privacy_level=item_data.privacy_level,
                tags=item_data.tags,
                original_id=item_data.original_id
            )

            # 应用隐私处理
            processed_item = privacy_guard.redact(memory_item, audience="system")
            memory_items.append(processed_item)

        # 存入向量存储
        vector_store.upsert(memory_items)
        return {"ok": True, "count": len(items), "item_ids": item_ids}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upsert failed: {str(e)}"
        )


@router.post("/search", response_model=QueryResponse)
def search_memory(
        request: QueryRequest,
        retriever: HybridRetriever = Depends(get_retriever),
        privacy_guard: PrivacyGuard = Depends(get_privacy_guard)
):
    """检索相关记忆项（带隐私过滤）"""
    try:
        # 执行混合检索
        results = retriever.retrieve(
            query=request.query,
            k=request.k,
            query_modalities=request.modalities
        )

        # 处理隐私和过滤
        hits = []
        for item, score in results:
            # 根据请求和受众过滤隐私内容
            audience = "clinician" if request.include_private else "default"
            redacted_item = privacy_guard.redact(item, audience=audience)

            # 转换为响应格式
            hits.append({
                "item": MemoryItemResponse.from_orm(redacted_item),
                "score": round(score, 4),
                "cosine_similarity": round(score * retriever.cfg.fuse_alpha, 4),
                "meta_score": round(score * (1 - retriever.cfg.fuse_alpha), 4)
            })

        return {
            "hits": hits,
            "retrieval_config": retriever.cfg
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/detect-conflicts", response_model=List[Dict[str, Any]])
def detect_conflicts(
        patient_id_hash: str,
        vector_store: VectorStore = Depends(get_vector_store),
        resolver: ConflictResolver = Depends(get_conflict_resolver)
):
    """检测特定患者的记忆冲突"""
    try:
        # 简化实现：实际应先检索该患者的所有记忆项
        # 这里模拟获取患者相关记忆
        patient_items = [item for item, _ in vector_store.search(
            query=f"patient_id:{patient_id_hash}", top_k=100
        )]

        # 检测冲突
        conflicts = resolver.detect(patient_items)

        # 格式化响应
        return [{
            "key": conflict.key,
            "item_ids": [item.id for item in conflict.items],
            "items": [MemoryItemResponse.from_orm(item) for item in conflict.items]
        } for conflict in conflicts]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conflict detection failed: {str(e)}"
        )


@router.post("/arbitrate", response_model=ArbitrationResponse)
def submit_arbitration(
        request: ArbitrationRequest,
        resolver: ConflictResolver = Depends(get_conflict_resolver),
        feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
):
    """提交仲裁结果（人工或自动）"""
    try:
        # 创建仲裁记录
        arb_record = ArbitrationRecord(
            arb_id=str(uuid4()),
            patient_id_hash=request.patient_id_hash,
            key=request.key,
            decided_item_id=request.decided_item_id,
            candidate_item_ids=request.candidate_item_ids,
            mode="human_resolved" if request.arbitrator else "auto",
            arbitrator=request.arbitrator,
            note=request.note
        )

        # 记录仲裁结果
        feedback_loop.log_arbitration(arb_record)

        # 如果有明确决策，更新相关记忆项的置信度
        if request.decided_item_id:
            # 对选中项增加置信度
            feedback_loop.update_memory_confidence(request.decided_item_id, delta=0.1)
            # 对未选中的候选项降低置信度
            for item_id in request.candidate_item_ids:
                if item_id != request.decided_item_id:
                    feedback_loop.update_memory_confidence(item_id, delta=-0.05)

        return {
            "arb_id": arb_record.arb_id,
            "status": "recorded",
            "timestamp": arb_record.timestamp
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Arbitration submission failed: {str(e)}"
        )


@router.post("/update-confidence", response_model=Dict[str, Any])
def update_memory_confidence(
        request: ConfidenceUpdateRequest,
        feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
):
    """更新记忆项的置信度"""
    success = feedback_loop.update_memory_confidence(
        item_id=request.item_id,
        delta=request.delta
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory item {request.item_id} not found or update failed"
        )

    return {
        "ok": True,
        "item_id": request.item_id,
        "delta": request.delta
    }


@router.get("/weights", response_model=RetrievalConfig)
def get_current_weights(
        feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
):
    """获取当前检索权重配置"""
    return feedback_loop.get_current_weights()


@router.post("/learn-weights", response_model=RetrievalConfig)
def trigger_weight_learning(
        feedback_loop: FeedbackLoop = Depends(get_feedback_loop)
):
    """触发权重学习（基于历史反馈）"""
    updated_weights = feedback_loop.learn_weights()
    return updated_weights
