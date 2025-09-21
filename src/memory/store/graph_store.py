from typing import Iterable, Dict, Any, List, Optional
from datetime import datetime
from neo4j import GraphDatabase, exceptions
from ..schemas import MemoryItem

class GraphStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化Neo4j图存储连接
        :param uri: Neo4j服务URI
        :param user: 用户名
        :param password: 密码
        :param database: 数据库名称
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._init_constraints()

    def _init_constraints(self) -> None:
        """初始化图数据库约束，确保唯一ID"""
        with self.driver.session(database=self.database) as session:
            # 创建患者节点唯一约束
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.id_hash IS UNIQUE
            """)
            # 创建事件节点唯一约束
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE
            """)

    def close(self) -> None:
        """关闭数据库连接"""
        self.driver.close()

    def upsert_events(self, items: Iterable[MemoryItem]) -> None:
        """
        将记忆项投影为时间线节点: (Patient)-[HAS_EVENT]->(Event)
        同时建立事件间的时序关系: (e1:Event)-[NEXT_EVENT]->(e2:Event)
        """
        with self.driver.session(database=self.database) as session:
            for item in items:
                # 1. 合并患者节点
                session.run("""
                    MERGE (p:Patient {id_hash: $patient_id_hash})
                    SET p.last_updated = $timestamp
                """, patient_id_hash=item.patient_id_hash, timestamp=datetime.utcnow().isoformat())

                # 2. 合并事件节点（包含所有属性）
                event_properties = {
                    "id": item.id,
                    "text": item.text,
                    "source": item.source,
                    "timestamp": item.timestamp.isoformat(),
                    "recency_days": item.recency_days,
                    "confidence": item.confidence,
                    "source_trust": item.source_trust,
                    "priority": item.priority,
                    "privacy_level": item.privacy_level,
                    "schema_version": item.schema_version
                }
                session.run("""
                    MERGE (e:Event {id: $id})
                    SET e += $properties
                """, id=item.id, properties=event_properties)

                # 3. 建立患者-事件关系
                session.run("""
                    MATCH (p:Patient {id_hash: $patient_id_hash})
                    MATCH (e:Event {id: $event_id})
                    MERGE (p)-[r:HAS_EVENT]->(e)
                    SET r.created_at = $now
                """, patient_id_hash=item.patient_id_hash, event_id=item.id, now=datetime.utcnow().isoformat())

                # 4. 添加事件标签（作为节点属性和关系）
                for tag in item.tags:
                    # 拆分标签键值对（如"tumor_status:progression"）
                    if ":" in tag:
                        key, value = tag.split(":", 1)
                        # 为事件添加标签属性
                        session.run("""
                            MATCH (e:Event {id: $event_id})
                            SET e[{key}] = $value
                        """, event_id=item.id, key=key, value=value)
                        # 创建标签节点并建立关系
                        session.run("""
                            MERGE (t:Tag {name: $tag})
                            MATCH (e:Event {id: $event_id})
                            MERGE (e)-[r:HAS_TAG]->(t)
                        """, tag=tag, event_id=item.id)

                # 5. 建立模态关系
                for modality in item.modalities:
                    session.run("""
                        MERGE (m:Modality {name: $modality})
                        MATCH (e:Event {id: $event_id})
                        MERGE (e)-[r:USES_MODALITY]->(m)
                    """, modality=modality, event_id=item.id)

            # 6. 为同一患者的事件建立时序关系
            self._create_timeline_relationships(session, [item.patient_id_hash for item in items])

    def _create_timeline_relationships(self, session, patient_ids: List[str]) -> None:
        """为每个患者的事件按时间排序并建立NEXT_EVENT关系"""
        for patient_id in set(patient_ids):  # 去重处理
            session.run("""
                MATCH (p:Patient {id_hash: $patient_id})-[:HAS_EVENT]->(e:Event)
                WITH p, e ORDER BY e.timestamp ASC
                WITH p, collect(e) as events
                UNWIND range(0, size(events)-2) as i
                WITH events[i] as e1, events[i+1] as e2
                MERGE (e1)-[r:NEXT_EVENT]->(e2)
                SET r.days_between = duration.between(
                    datetime(e1.timestamp), 
                    datetime(e2.timestamp)
                ).days
            """, patient_id=patient_id)

    def path_timeline(self, patient_id_hash: str,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        返回患者的时序事件列表（手术、化疗、影像复查、标志物等）
        :param patient_id_hash: 患者哈希ID
        :param start_date: 起始日期（可选）
        :param end_date: 结束日期（可选）
        :return: 按时间排序的事件列表
        """
        query = """
            MATCH (p:Patient {id_hash: $patient_id})-[:HAS_EVENT]->(e:Event)
            WHERE 1=1
        """
        params = {"patient_id": patient_id_hash}

        # 添加日期过滤条件
        if start_date:
            query += " AND e.timestamp >= $start_date"
            params["start_date"] = start_date.isoformat()
        if end_date:
            query += " AND e.timestamp <= $end_date"
            params["end_date"] = end_date.isoformat()

        # 完成查询并按时间排序
        query += """
            OPTIONAL MATCH (e)-[:HAS_TAG]->(t:Tag)
            OPTIONAL MATCH (e)-[:USES_MODALITY]->(m:Modality)
            WITH e, collect(DISTINCT t.name) as tags, collect(DISTINCT m.name) as modalities
            ORDER BY e.timestamp ASC
            RETURN 
                e.id as event_id,
                e.text as text,
                e.source as source,
                e.timestamp as timestamp,
                e.recency_days as recency_days,
                e.confidence as confidence,
                e.source_trust as source_trust,
                e.priority as priority,
                tags,
                modalities
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def get_related_events(self, event_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        获取与指定事件相关的事件（通过标签关联）
        :param event_id: 事件ID
        :param max_depth: 关联深度
        :return: 相关事件列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (e:Event {id: $event_id})-[:HAS_TAG]->(t:Tag)<-[:HAS_TAG]-(related:Event)
                WHERE related.id <> $event_id
                WITH related, count(t) as common_tags
                OPTIONAL MATCH (related)-[:HAS_TAG]->(rt:Tag)
                OPTIONAL MATCH (related)-[:USES_MODALITY]->(rm:Modality)
                RETURN 
                    related.id as event_id,
                    related.text as text,
                    related.timestamp as timestamp,
                    related.source as source,
                    common_tags,
                    collect(DISTINCT rt.name) as tags,
                    collect(DISTINCT rm.name) as modalities
                ORDER BY common_tags DESC, related.timestamp DESC
            """, event_id=event_id)
            return [record.data() for record in result]

    def delete_patient_data(self, patient_id_hash: str) -> None:
        """删除患者的所有相关数据（用于隐私合规）"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (p:Patient {id_hash: $patient_id})-[:HAS_EVENT]->(e:Event)
                DETACH DELETE e
                DELETE p
            """, patient_id=patient_id_hash)