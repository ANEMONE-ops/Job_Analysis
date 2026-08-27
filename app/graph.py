"""版本化时序知识图谱。

以 JSON 文件持久化，NetworkX 内存图提供查询。
- 节点：Role / Skill / Responsibility / Industry / Evidence / Resume / Course
- 边：requires / preferred_requires / belongs_to / prerequisite / applied_in /
      substitutes / evolves_to，携带 valid_from / valid_to / confidence / trust_score。
- 版本化：每次写入生成版本号，支持时间切片与版本回滚。
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from .models import Claim, TrendInfo


class TemporalGraph:
    def __init__(self, store_dir: Path | str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.versions_path = self.store_dir / "versions"
        self.versions_path.mkdir(exist_ok=True)
        self.snapshot_path = self.store_dir / "graph.json"
        self.G = nx.MultiDiGraph()
        self.version = 0
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        if self.snapshot_path.exists():
            try:
                data = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
                self.version = data.get("version", 0)
                self.G = nx.node_link_graph(data, edges="links")
                return
            except (json.JSONDecodeError, ValueError, KeyError):
                # 快照损坏（如写入被中断导致截断）：退化为空图，由上层重建
                pass
        self._snapshot()

    def _snapshot(self) -> None:
        data = nx.node_link_data(self.G, edges="links")
        data["version"] = self.version
        self.snapshot_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def commit(self, message: str = "") -> int:
        """保存当前图并生成新版本。"""
        self.version += 1
        # 保存版本快照
        version_file = self.versions_path / f"v{self.version:04d}.json"
        data = nx.node_link_data(self.G, edges="links")
        data["version"] = self.version
        data["commit_message"] = message
        data["committed_at"] = datetime.now().isoformat()
        version_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._snapshot()
        return self.version

    def clear(self) -> None:
        """清空图（保留版本号），用于重建。"""
        self.G = nx.MultiDiGraph()

    def rollback(self, version: int) -> bool:
        version_file = self.versions_path / f"v{version:04d}.json"
        if not version_file.exists():
            return False
        data = json.loads(version_file.read_text(encoding="utf-8"))
        self.G = nx.node_link_graph(data, edges="links")
        self.version = version
        self._snapshot()
        return True

    def list_versions(self) -> list[dict]:
        out = []
        for f in sorted(self.versions_path.glob("v*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                out.append({
                    "version": data.get("version"),
                    "commit_message": data.get("commit_message", ""),
                    "committed_at": data.get("committed_at", ""),
                })
            except Exception:
                continue
        return out

    # ---------- 写入 ----------
    def upsert_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        self.G.add_node(node_id, type=node_type, **attrs)

    def add_edge(self, u: str, v: str, rel: str, **attrs: Any) -> None:
        self.G.add_edge(u, v, relation=rel, **attrs)

    def add_claim(self, claim: Claim) -> None:
        """将 Claim 落到图谱：subject -[predicate]-> object。"""
        self.upsert_node(claim.subject, self._type_of(claim.subject), name=self._display_name(claim.subject))
        self.upsert_node(claim.object, self._type_of(claim.object), name=self._display_name(claim.object))
        self.add_edge(
            claim.subject,
            claim.object,
            claim.predicate,
            source_id=claim.source_id,
            evidence_span=claim.evidence_span,
            observed_at=claim.observed_at,
            valid_from=claim.valid_from,
            valid_to=claim.valid_to,
            trust_score=claim.trust_score,
            confidence=claim.confidence,
            verification_status=claim.verification_status,
            review_status=claim.review_status,
            is_required=claim.is_required,
            proficiency=claim.proficiency,
        )

    def _type_of(self, node_id: str) -> str:
        for prefix, t in [("role:", "Role"), ("skill:", "Skill"), ("resp:", "Responsibility"),
                          ("industry:", "Industry"), ("evid:", "Evidence"), ("resume:", "Resume"),
                          ("course:", "Course")]:
            if node_id.startswith(prefix):
                return t
        return "Unknown"

    @staticmethod
    def _display_name(node_id: str) -> str:
        """从 node id 中还原可读名称（id 形如 skill:Python）。"""
        if ":" in node_id:
            return node_id.split(":", 1)[1]
        return node_id

    # ---------- 查询 ----------
    def role_skills(self, role_id: str, valid_at: Optional[str] = None) -> dict[str, list[dict]]:
        """返回某岗位的必备/加分技能（按 valid_from/valid_to 过滤）。"""
        out = {"required": [], "preferred": []}
        if role_id not in self.G:
            return out
        for _, v, data in self.G.out_edges(role_id, data=True):
            if data.get("relation") not in ("requires", "preferred_requires"):
                continue
            if not self._valid(data, valid_at):
                continue
            item = {"skill": v, "name": self.G.nodes[v].get("name") or self._display_name(v), **data}
            out["required" if data.get("is_required", True) else "preferred"].append(item)
        return out

    def _valid(self, edge: dict, valid_at: Optional[str]) -> bool:
        if valid_at is None:
            return True
        vf, vt = edge.get("valid_from"), edge.get("valid_to")
        if vf and vf > valid_at:
            return False
        if vt and vt < valid_at:
            return False
        return True

    def skill_trend(self, skill_id: str, window_days: int = 180) -> TrendInfo:
        """统计某技能在时间窗内的出现与增长率，判定生命周期状态。"""
        mentions: list[str] = []
        if skill_id in self.G:
            mentions.append(self.G.nodes[skill_id].get("first_seen", ""))
            mentions.append(self.G.nodes[skill_id].get("last_seen", ""))
        first = min([m for m in mentions if m], default="")
        last = max([m for m in mentions if m], default="")
        # 出现次数来自入边（多个岗位提到该技能）
        count = 0
        for _, _, data in self.G.in_edges(skill_id, data=True):
            if data.get("relation") in ("requires", "preferred_requires"):
                count += 1
        growth = count / max(1, self.G.number_of_nodes() / 20) if count else 0.0
        if count == 0:
            state = "萌芽"
        elif growth > 1.0:
            state = "增长"
        else:
            state = "稳定"
        return TrendInfo(
            skill=skill_id,
            trend_state=state,
            growth_rate=round(growth, 3),
            emergence_score=round(min(1.0, growth), 3),
            first_seen=first,
            last_seen=last,
        )

    def subgraph_by_filter(self, industry: str = "", tech_stack: str = "", level: str = "") -> dict:
        """按行业/技术栈/级别过滤，返回可渲染的节点-边结构（全景图谱）。

        行业/级别过滤作用于 Role 节点；Skill 节点始终保留，边按保留节点过滤。
        """
        nodes, edges = [], []
        keep: set[str] = set()
        for n, data in self.G.nodes(data=True):
            if data.get("type") == "Role":
                if industry and data.get("industry") != industry:
                    continue
                if level and data.get("level") != level:
                    continue
            elif data.get("type") == "Skill":
                pass  # 技能节点始终保留
            else:
                continue
            keep.add(n)
            nodes.append({"id": n, "label": data.get("name", n), "type": data.get("type"),
                          "category": data.get("category", ""),
                          "level": data.get("level", ""),
                          "industry": data.get("industry", ""),
                          "occurrence_count": data.get("occurrence_count", 0),
                          "first_seen": data.get("first_seen", ""),
                          "last_seen": data.get("last_seen", "")})
        for u, v, data in self.G.edges(data=True):
            if u not in keep or v not in keep:
                continue
            edges.append({"source": u, "target": v, "relation": data.get("relation"),
                          "trust_score": data.get("trust_score", 0),
                          "is_required": data.get("is_required", True)})

        # 节点可信度：取关联边 trust_score 均值；无关联边时给中性值 0.5。
        # trust_score 本是边的属性（来源可信度），节点本身没有，这里聚合展示。
        node_ts: dict[str, list[float]] = {}
        for e in edges:
            node_ts.setdefault(e["source"], []).append(e["trust_score"])
            node_ts.setdefault(e["target"], []).append(e["trust_score"])
        for n in nodes:
            scores = node_ts.get(n["id"], [])
            if scores:
                ts = sum(scores) / len(scores)
            elif n["type"] == "Role":
                ts = 0.85  # 岗位节点默认高可信（来源于官方/招聘平台）
            else:
                ts = 0.5   # 技能节点中性
            n["trust_score"] = round(ts, 4)
        return {"nodes": nodes, "edges": edges}

    def to_dict(self) -> dict:
        return {"nodes": [
            {"id": n, **{k: v for k, v in d.items() if k != "type"},
             "type": d.get("type")} for n, d in self.G.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.G.edges(data=True)]}
