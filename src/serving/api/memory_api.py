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