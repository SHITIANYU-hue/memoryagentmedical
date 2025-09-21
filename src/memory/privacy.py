from typing import Dict, Any, List, Optional
import re
from datetime import datetime
from cryptography.fernet import Fernet
from ..schemas import MemoryItem, PrivacyLevel


class PrivacyGuard:
    def __init__(self, policy: Dict[str, Any]):
        """
        初始化隐私保护组件
        :param policy: 隐私策略配置，来自configs/memory.yaml
        """
        self.policy = policy
        self.encryption_keys = {
            "high": Fernet.generate_key(),
            "medium": Fernet.generate_key()
        }
        self.fernet_high = Fernet(self.encryption_keys["high"])
        self.fernet_medium = Fernet(self.encryption_keys["medium"])

        # PII模式匹配规则
        self.pii_patterns = {
            "name": re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+"),
            "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "mrn": re.compile(r"MRN:\s*\d{6,10}"),  # 医疗记录号
            "dob": re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")  # 出生日期
        }

        # 敏感医疗信息模式
        self.medical_sensitive_patterns = {
            "drug_dose": re.compile(r"\b\d+[mg|ml|g|unit|IU]\/?[kg|day|hr]?\b"),  # 药物剂量
            "genetic_info": re.compile(r"[A-Z0-9]+\s*mutation|variant|allele", re.IGNORECASE),  # 基因信息
            "hiv_status": re.compile(r"HIV\s*(positive|negative|status)", re.IGNORECASE),
            "mental_health": re.compile(r"depression|schizophrenia|bipolar|suicide", re.IGNORECASE)
        }

    def redact(self, item: MemoryItem, audience: str) -> MemoryItem:
        """
        根据受众和隐私级别处理记忆项
        :param item: 原始记忆项
        :param audience: 受众类型 (patient/clinician/research)
        :return: 处理后的记忆项
        """
        # 复制原始项避免修改源数据
        redacted_item = MemoryItem(**item.__dict__)

        # 应用隐私策略
        policy = self.policy.get("privacy", {}).get(item.privacy_level, {})

        # 高隐私级别处理
        if item.privacy_level == "high":
            if audience != "clinician":
                # 非临床医生只能看到摘要信息
                redacted_item.text = self._generate_summary(redacted_item.text)
                redacted_item.embedding = []  # 隐藏嵌入向量
                if policy.get("encrypt", False):
                    redacted_item.text = self._encrypt(redacted_item.text, "high")

        # 中隐私级别处理
        elif item.privacy_level == "medium":
            if audience == "research":
                # 研究者看不到具体标识符
                redacted_item = self._remove_pii(redacted_item)
                redacted_item.embedding = []
            if policy.get("encrypt", False):
                redacted_item.text = self._encrypt(redacted_item.text, "medium")

        # 低隐私级别处理 - 仅移除PII
        else:
            redacted_item = self._remove_pii(redacted_item)

        return redacted_item

    def check_output(self, text: str, level: PrivacyLevel) -> str:
        """
        出口审查：检查并过滤输出文本中的敏感信息
        :param text: 待检查文本
        :param level: 隐私级别
        :return: 过滤后的文本
        """
        # 移除所有PII
        filtered = self._replace_patterns(text, self.pii_patterns, "[REDACTED]")

        # 根据隐私级别过滤医疗敏感信息
        if level in ["high", "medium"]:
            filtered = self._replace_patterns(filtered, self.medical_sensitive_patterns, "[SENSITIVE]")

        # 高隐私级别额外过滤治疗方案细节
        if level == "high":
            filtered = self._filter_treatment_details(filtered)

        return filtered

    def _remove_pii(self, item: MemoryItem) -> MemoryItem:
        """移除个人身份信息"""
        item.text = self._replace_patterns(item.text, self.pii_patterns, "[PII_REDACTED]")
        return item

    def _encrypt(self, text: str, level: str) -> str:
        """根据隐私级别加密文本"""
        if level == "high":
            return self.fernet_high.encrypt(text.encode()).decode()
        elif level == "medium":
            return self.fernet_medium.encrypt(text.encode()).decode()
        return text

    def decrypt(self, encrypted_text: str, level: str) -> str:
        """解密文本（仅授权用户使用）"""
        try:
            if level == "high":
                return self.fernet_high.decrypt(encrypted_text.encode()).decode()
            elif level == "medium":
                return self.fernet_medium.decrypt(encrypted_text.encode()).decode()
        except:
            return "[DECRYPTION_FAILED]"
        return encrypted_text

    def _generate_summary(self, text: str) -> str:
        """为高隐私级别内容生成摘要（实际应用中可替换为LLM生成）"""
        # 简单摘要实现，实际中应使用更复杂的NLP模型
        sentences = re.split(r'[.!?]', text)
        if len(sentences) >= 3:
            return f"[SUMMARY] {sentences[0]}. {sentences[1]}..."
        return f"[SUMMARY] {text[:100]}..."

    def _replace_patterns(self, text: str, patterns: Dict[str, re.Pattern], replacement: str) -> str:
        """替换文本中匹配模式的内容"""
        result = text
        for name, pattern in patterns.items():
            result = pattern.sub(replacement, result)
        return result

    def _filter_treatment_details(self, text: str) -> str:
        """过滤高隐私级别内容中的治疗细节"""
        treatment_keywords = [
            "chemotherapy", "radiation", "surgery", "dose",
            "prescription", "medication", "treatment plan"
        ]
        for keyword in treatment_keywords:
            text = re.sub(rf"\b{keyword}\b.*?\.", f"[TREATMENT_DETAILS_REDACTED].", text,
                          flags=re.IGNORECASE | re.DOTALL)
        return text

    def audit_access(self, item_id: str, user: str, action: str) -> None:
        """记录访问审计日志"""
        # 实际应用中应写入安全日志系统
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "item_id": item_id,
            "user": user,
            "action": action,
            "status": "success"
        }
        print(f"AUDIT: {audit_entry}")  # 替换为实际日志记录