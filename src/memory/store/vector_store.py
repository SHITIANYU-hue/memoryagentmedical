from typing import List, Tuple, Optional, Dict
import faiss
import numpy as np
from datetime import datetime
from ..schemas import MemoryItem


class VectorStore:
    def upsert(self, items: List[MemoryItem]):
        """插入或更新记忆项到向量存储"""
        raise NotImplementedError

    def search(self, query: str, top_k: int = 16) -> List[Tuple[MemoryItem, float]]:
        """
        检索与查询相关的记忆项
        返回: (记忆项, 余弦相似度) 列表
        """
        raise NotImplementedError

    def get_by_ids(self, ids: List[str]) -> Dict[str, MemoryItem]:
        """通过ID批量获取记忆项"""
        raise NotImplementedError


class FAISSStore(VectorStore):
    def __init__(self, dim: int = 1024, index_path: Optional[str] = None):
        """
        初始化FAISS向量存储
        :param dim: 嵌入向量维度
        :param index_path: 预训练索引路径(可选)
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)  # 使用L2距离(可转换为余弦相似度)
        self.id_to_item: Dict[str, MemoryItem] = {}  # 存储原始数据
        self.id_to_index: Dict[str, int] = {}  # 映射item_id到索引位置

        if index_path:
            self.load(index_path)

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """归一化向量以支持余弦相似度计算"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def upsert(self, items: List[MemoryItem]):
        """插入或更新记忆项"""
        new_vectors = []
        new_ids = []

        for item in items:
            # 检查是否需要更新
            if item.id in self.id_to_index:
                idx = self.id_to_index[item.id]
                # 更新向量
                vec = np.array(item.embedding).reshape(1, -1).astype(np.float32)
                self.index.reconstruct_n(idx, 1, vec)
                self.id_to_item[item.id] = item
            else:
                # 新增项目
                new_ids.append(item.id)
                new_vectors.append(item.embedding)
                self.id_to_item[item.id] = item

        if new_vectors:
            # 批量添加新向量
            vec_array = np.array(new_vectors).astype(np.float32)
            vec_array = self._normalize_vectors(vec_array)  # 归一化
            start_idx = self.index.ntotal
            self.index.add(vec_array)

            # 更新ID映射
            for i, item_id in enumerate(new_ids):
                self.id_to_index[item_id] = start_idx + i

    def search(self, query_embedding: List[float], top_k: int = 16) -> List[Tuple[MemoryItem, float]]:
        """
        根据查询向量检索相似项
        :param query_embedding: 查询文本的嵌入向量
        :param top_k: 返回的最大结果数
        """
        if not self.id_to_item:
            return []

        # 处理查询向量
        query_vec = np.array(query_embedding).reshape(1, -1).astype(np.float32)
        query_vec = self._normalize_vectors(query_vec)

        # 执行搜索(L2距离)
        distances, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        # 转换L2距离为余弦相似度(归一化向量空间中)
        # 余弦相似度 = 1 - (L2距离^2)/2
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS返回-1表示无结果
                continue
            item_id = next(k for k, v in self.id_to_index.items() if v == idx)
            item = self.id_to_item.get(item_id)
            if item:
                cos_sim = 1.0 - (dist ** 2) / 2.0
                results.append((item, float(cos_sim)))

        return results

    def get_by_ids(self, ids: List[str]) -> Dict[str, MemoryItem]:
        """通过ID批量获取记忆项"""
        return {id_: self.id_to_item[id_] for id_ in ids if id_ in self.id_to_item}

    def save(self, index_path: str):
        """保存索引到磁盘"""
        faiss.write_index(self.index, f"{index_path}_index.faiss")
        # 实际应用中应使用更高效的序列化方式(如pickle/msgpack)
        import pickle
        with open(f"{index_path}_metadata.pkl", "wb") as f:
            pickle.dump({
                "id_to_item": self.id_to_item,
                "id_to_index": self.id_to_index,
                "dim": self.dim
            }, f)

    def load(self, index_path: str):
        """从磁盘加载索引"""
        self.index = faiss.read_index(f"{index_path}_index.faiss")
        import pickle
        with open(f"{index_path}_metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.id_to_item = data["id_to_item"]
            self.id_to_index = data["id_to_index"]
            self.dim = data["dim"]