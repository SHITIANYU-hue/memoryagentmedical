from typing import Dict, Any, List, Optional
from datetime import datetime
from .schemas import ArbitrationRecord, MemoryItem, RetrievalConfig
from .store.vector_store import VectorStore
import numpy as np
from sklearn.metrics import precision_score, recall_score


class FeedbackLoop:
    def __init__(self, vector_store: VectorStore, initial_weights: Dict[str, float]):
        """
        初始化反馈循环组件
        :param vector_store: 向量存储实例，用于更新记忆项
        :param initial_weights: 初始权重配置 {w_recency, w_trust, w_priority, w_modality}
        """
        self.vector_store = vector_store
        self.weights = initial_weights.copy()
        self.arbitration_history: List[ArbitrationRecord] = []
        self.retrieval_metrics: List[Dict[str, Any]] = []

        # 权重调整参数
        self.learning_rate = 0.05
        self.weight_bounds = (0.05, 0.85)  # 权重上下限

    def log_arbitration(self, record: ArbitrationRecord) -> None:
        """
        记录仲裁结果到历史记录
        :param record: 仲裁记录
        """
        self.arbitration_history.append(record)
        # 实际应用中应持久化存储（如数据库）
        if len(self.arbitration_history) % 100 == 0:
            self._persist_arbitration_history()

    def log_retrieval_feedback(self,
                               query: str,
                               retrieved_items: List[MemoryItem],
                               relevant_ids: List[str],  # 人工标注的相关记忆ID
                               precision: Optional[float] = None,
                               recall: Optional[float] = None) -> None:
        """
        记录检索反馈数据
        :param query: 查询文本
        :param retrieved_items: 检索到的记忆项
        :param relevant_ids: 实际相关的记忆项ID
        :param precision: 可选，预计算的精确率
        :param recall: 可选，预计算的召回率
        """
        retrieved_ids = [item.id for item in retrieved_items]

        # 自动计算精确率和召回率（如果未提供）
        if precision is None:
            y_true = [1 if id in relevant_ids else 0 for id in retrieved_ids]
            y_pred = [1] * len(retrieved_ids)
            precision = precision_score(y_true, y_pred, zero_division=0)

        if recall is None:
            y_true = [1 if id in retrieved_ids else 0 for id in relevant_ids]
            y_pred = [1] * len(relevant_ids)
            recall = recall_score(y_true, y_pred, zero_division=0)

        metric_entry = {
            "timestamp": datetime.utcnow(),
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "precision": precision,
            "recall": recall,
            "weights_used": self.weights.copy()
        }
        self.retrieval_metrics.append(metric_entry)

    def learn_weights(self) -> Dict[str, float]:
        """
        基于历史反馈更新权重（结合检索性能和仲裁结果）
        :return: 更新后的权重
        """
        if not self.retrieval_metrics and not self.arbitration_history:
            return self.weights.copy()  # 无反馈时返回当前权重

        # 1. 计算检索性能导向的权重调整
        retrieval_signals = self._calculate_retrieval_signals()

        # 2. 计算仲裁结果导向的权重调整
        arbitration_signals = self._calculate_arbitration_signals()

        # 3. 融合信号并更新权重
        for weight_name in self.weights:
            # 检索信号权重更高（0.7），仲裁信号辅助调整（0.3）
            delta = (0.7 * retrieval_signals.get(weight_name, 0.0) +
                     0.3 * arbitration_signals.get(weight_name, 0.0))

            # 应用学习率
            adjusted_delta = delta * self.learning_rate

            # 更新权重并裁剪到合理范围
            new_weight = self.weights[weight_name] + adjusted_delta
            self.weights[weight_name] = max(
                self.weight_bounds[0],
                min(new_weight, self.weight_bounds[1])
            )

        return self.weights.copy()

    def _calculate_retrieval_signals(self) -> Dict[str, float]:
        """计算基于检索性能的权重调整信号"""
        if len(self.retrieval_metrics) < 5:  # 至少需要5个样本才有统计意义
            return {k: 0.0 for k in self.weights}

        # 取最近的20个检索记录
        recent_metrics = self.retrieval_metrics[-20:]
        avg_precision = np.mean([m["precision"] for m in recent_metrics])
        target_precision = 0.8  # 目标精确率

        # 计算各权重与检索性能的相关性（简化版）
        signals = {}
        for weight_name in self.weights:
            # 分析权重值与精确率的关系
            weight_values = [m["weights_used"][weight_name] for m in recent_metrics]
            precisions = [m["precision"] for m in recent_metrics]

            # 计算相关性（简化为协方差）
            covariance = np.cov(weight_values, precisions)[0][1]

            # 根据当前性能与目标的差距调整信号方向
            if avg_precision < target_precision:
                # 性能不足时，增强正相关权重，减弱负相关权重
                signals[weight_name] = covariance
            else:
                # 性能达标时，微调保持稳定
                signals[weight_name] = -0.1 * covariance

        return signals

    def _calculate_arbitration_signals(self) -> Dict[str, float]:
        """计算基于仲裁结果的权重调整信号"""
        if len(self.arbitration_history) < 3:
            return {k: 0.0 for k in self.weights}

        # 分析最近的仲裁记录
        recent_arb = self.arbitration_history[-10:]
        human_intervention_rate = np.mean([
            1 if arb.mode == "human_required" or arb.mode == "human_resolved" else 0
            for arb in recent_arb
        ])

        signals = {}
        # 人工介入率过高时，调整信任度权重
        if human_intervention_rate > 0.3:  # 超过30%人工介入需要调整
            # 增加来源信任度权重，减少其他权重
            signals["w_trust"] = 0.2
            signals["w_recency"] = -0.1
            signals["w_priority"] = -0.1
            signals["w_modality"] = 0.0
        elif human_intervention_rate < 0.1:  # 人工介入率过低，可适当降低信任权重
            signals["w_trust"] = -0.1
            signals["w_recency"] = 0.05
            signals["w_priority"] = 0.05
            signals["w_modality"] = 0.0

        return signals

    def update_memory_confidence(self, item_id: str, delta: float) -> bool:
        """
        更新记忆项的置信度
        :param item_id: 记忆项ID
        :param delta: 置信度变化量（可正可负）
        :return: 是否更新成功
        """
        # 实际实现需要从向量存储中获取并更新记忆项
        # 这里简化为模拟实现
        try:
            # 伪代码：从存储中获取项目
            # item = self.vector_store.get_item(item_id)
            # item.confidence = max(0.0, min(1.0, item.confidence + delta))
            # self.vector_store.upsert([item])
            return True
        except Exception as e:
            print(f"Failed to update memory confidence: {e}")
            return False

    def _persist_arbitration_history(self) -> None:
        """持久化仲裁历史（实际应用中应实现）"""
        # 示例实现：写入数据库或文件
        print(f"Persisting {len(self.arbitration_history)} arbitration records")
        # with open("arbitration_history.jsonl", "a") as f:
        #     for record in self.arbitration_history:
        #         f.write(json.dumps(asdict(record)) + "\n")

    def get_current_weights(self) -> RetrievalConfig:
        """获取当前权重配置（转换为RetrievalConfig对象）"""
        return RetrievalConfig(
            w_recency=self.weights["w_recency"],
            w_trust=self.weights["w_trust"],
            w_priority=self.weights["w_priority"],
            w_modality=self.weights["w_modality"],
            fuse_alpha=self.weights.get("fuse_alpha", 0.5)
        )