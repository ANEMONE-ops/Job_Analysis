"""智能问答模块：基于正则的意图匹配，不依赖外部 LLM API。

设计思路：
- 用一组正则规则把用户问题分类到固定意图（问候/帮助/岗位技能/
  技能反查岗位/数据规模/评测指标/新岗位/岗位列表/匹配引导等）；
- 命中意图后直接查询本地知识图谱（Pipeline）生成答案，
  保证每个回答都有真实图谱数据支撑（可追溯、零幻觉）；
- 未命中任何意图时返回引导性建议而不是编造答案。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .config import load_config
from .skills import SKILL_DICT, normalize_skill

# ---------- 意图正则表 ----------
# 顺序即优先级：越靠前越优先匹配。
_INTENT_PATTERNS = [
    ("greeting", re.compile(r"^\s*(你好|您好|hi|hello|嗨|哈喽|在吗|在不在)", re.I)),
    ("help", re.compile(r"(帮助|帮我看看怎么用|你能(做|干)(什么|啥)|你会(什么|啥)|怎么用|功能介绍|使用说明|能问(什么|啥))", re.I)),
    ("thanks", re.compile(r"(谢谢|感谢|辛苦了|thanks|thank you)", re.I)),
    ("metrics", re.compile(r"(准确率|评测|指标|幻觉率|可信度|f1|precision|recall|达标|成绩|表现如何|效果如何)", re.I)),
    ("scale", re.compile(r"(多少.*(岗位|技能|节点|边|数据|简历|jd)|数据规模|规模有(多大|多少)|多少条)", re.I)),
    ("new_roles", re.compile(r"(新(兴|的)?岗位|新兴职业|萌芽|有什么新|趋势|新出现)", re.I)),
    ("role_list", re.compile(r"(有哪些岗位|岗位列表|什么岗位|岗位(种类|都有(哪些|什么))|列出.*岗位)", re.I)),
    ("industry", re.compile(r"(哪些行业|什么行业|行业覆盖|覆盖.*行业)", re.I)),
    ("match_guide", re.compile(r"(简历|匹配|差距|学习路径|怎么用.*匹配|上传)", re.I)),
]

# 岗位→技能问题在 answer() 中通过「已知岗位名 + 意图动词」联合判断。
# 技能→岗位问题：「哪些岗位需要XX」
_SKILLQ_PAT = re.compile(r"(哪些|什么|什么样).{0,4}?(岗位|工作|职业|职位).{0,10}?(需要|用到|涉及|要求|会|要)")


def answer(question: str, pipeline, config=None) -> dict:
    """问答主入口：正则匹配意图 -> 查询图谱 -> 生成答案。"""
    text = (question or "").strip()
    config = config or load_config()

    if not text:
        return _wrap("fallback", "请输入你的问题，例如：「人工智能算法工程师需要什么技能？」")

    # 1. 问候
    if _INTENT_PATTERNS[0][1].search(text):
        return _wrap("greeting",
                     "你好！我是岗位能力图谱助手 🤖 可以问我：\n"
                     "· 「人工智能算法工程师需要什么技能？」\n"
                     "· 「哪些岗位需要 Python？」\n"
                     "· 「系统评测指标怎么样？」\n"
                     "· 「图谱里有多少数据？」")

    # 2. 帮助
    if _INTENT_PATTERNS[1][1].search(text):
        return _wrap("help",
                     "我可以基于本地岗位能力图谱回答这些问题：\n"
                     "1️⃣ 岗位→技能：「XX岗位需要什么技能？」\n"
                     "2️⃣ 技能→岗位：「哪些岗位需要 Machine Learning？」\n"
                     "3️⃣ 岗位列表：「系统里有哪些岗位？」\n"
                     "4️⃣ 新兴岗位：「有什么新兴岗位？」\n"
                     "5️⃣ 数据规模：「图谱里有多少数据？」\n"
                     "6️⃣ 评测指标：「系统的匹配准确率是多少？」\n"
                     "7️⃣ 匹配引导：「我想做简历匹配该怎么做？」")

    # 3. 感谢
    if _INTENT_PATTERNS[2][1].search(text):
        return _wrap("thanks", "不客气！还有其他关于岗位能力的问题欢迎继续提问～")

    # 4. 技能→岗位（先于岗位→技能判断，避免「哪些岗位需要」被误判为岗位问题）
    m = _SKILLQ_PAT.search(text)
    skill_hit = _find_skill_in_text(text)
    if m and skill_hit:
        return _answer_skill_to_roles(pipeline, skill_hit)
    if skill_hit and re.search(r"(岗位|工作|职业|职位)", text) and re.search(r"(哪些|哪些|什么|谁)", text):
        return _answer_skill_to_roles(pipeline, skill_hit)

    # 5. 岗位→技能：在问题里找到已知岗位名
    role_hit = _find_role_in_text(text, pipeline)
    if role_hit and re.search(
            r"(需要|要|要求|学什么|会什么|掌握|具备|技能|职责|做什么|干什么|负责)", text):
        return _answer_role_skills(pipeline, role_hit, config)

    # 6. 数据规模
    if _INTENT_PATTERNS[4][1].search(text):
        return _answer_scale(pipeline)

    # 7. 评测指标
    if _INTENT_PATTERNS[3][1].search(text):
        return _answer_metrics(config)

    # 8. 新岗位
    if _INTENT_PATTERNS[5][1].search(text):
        return _answer_new_roles(pipeline)

    # 9. 岗位列表
    if _INTENT_PATTERNS[6][1].search(text) or role_hit:
        return _answer_role_list(pipeline)

    # 10. 行业
    if _INTENT_PATTERNS[7][1].search(text):
        return _answer_industries(pipeline)

    # 11. 匹配引导
    if _INTENT_PATTERNS[8][1].search(text):
        return _wrap("match_guide",
                     "做人岗匹配很简单：\n"
                     "1️⃣ 在左侧导航进入「个人简历」页，粘贴简历文本或上传 PDF/Word；\n"
                     "2️⃣ 进入「人岗匹配」页选择目标岗位；\n"
                     "3️⃣ 系统会输出匹配度、维度得分、技能差距和学习路径。\n"
                     "也可以直接问我「XX岗位需要什么技能」先了解岗位要求～")

    # 12. 兜底：不编造，给出建议
    return _wrap("fallback",
                 "这个问题我暂时没听懂 🤔 我擅长回答岗位能力相关的问题，你可以试试：\n"
                 "· 「人工智能算法工程师需要什么技能？」\n"
                 "· 「哪些岗位需要 Python？」\n"
                 "· 「系统评测指标怎么样？」",
                 )


# ---------- 意图答案生成 ----------

def _find_role_in_text(text: str, pipeline) -> Optional[str]:
    """在问题中查找已知岗位名（最长匹配优先）。"""
    roles = pipeline.known_roles()
    hit = None
    for r in roles:
        if r and r in text:
            if hit is None or len(r) > len(hit):
                hit = r
    return hit


def _find_skill_in_text(text: str) -> Optional[str]:
    """在问题中查找技能本体里的技能名或别名。"""
    lowered = text.lower()
    best = None
    for name, meta in SKILL_DICT.items():
        cands = [name.lower()] + [a.lower() for a in meta.get("aliases", [])]
        for c in cands:
            if c and c in lowered:
                if best is None or len(c) > len(best):
                    best = c
    if best:
        # 还原为规范技能名
        return normalize_skill(best) or best
    return None


def _answer_role_skills(pipeline, role_name: str, config) -> dict:
    required, preferred = pipeline._role_skill_sets(role_name)
    req_names = [s["skill"] for s in required]
    pref_names = [s["skill"] for s in preferred]
    lines = [f"💼 **{role_name}** 的能力要求（来自 {len(pipeline._jobs)} 条 JD 的图谱聚合）："]
    if req_names:
        lines.append("\n**核心必备技能**：" + "、".join(req_names))
    else:
        lines.append("\n**核心必备技能**：图谱中暂无高置信度的必备技能记录")
    if pref_names:
        lines.append("\n**加分技能**：" + "、".join(pref_names))
    lines.append("\n\n💡 想看自己和这个岗位的差距，可以到「人岗匹配」页上传简历做诊断。")
    return _wrap("role_skills", "\n".join(lines),
                 extra={"role_name": role_name,
                        "required_skills": req_names, "preferred_skills": pref_names})


def _answer_skill_to_roles(pipeline, skill: str) -> dict:
    roles_with = []
    for role in pipeline.known_roles():
        required, preferred = pipeline._role_skill_sets(role)
        req_names = [s["skill"] for s in required]
        pref_names = [s["skill"] for s in preferred]
        if skill in req_names:
            roles_with.append((role, "必备"))
        elif skill in pref_names:
            roles_with.append((role, "加分"))
    if not roles_with:
        return _wrap("skill_roles",
                     f"图谱中暂未发现把「{skill}」作为核心要求的岗位。\n"
                     f"你可以换个技能名问问，或者到「知识库」页浏览完整技能本体。")
    req_roles = [r for r, t in roles_with if t == "必备"]
    pref_roles = [r for r, t in roles_with if t == "加分"]
    lines = [f"🔍 技能 **{skill}** 在图谱中的岗位分布："]
    if req_roles:
        lines.append("\n**作为必备技能**：" + "、".join(req_roles))
    if pref_roles:
        lines.append("\n**作为加分技能**：" + "、".join(pref_roles))
    return _wrap("skill_roles", "\n".join(lines),
                 extra={"skill": skill,
                        "required_roles": req_roles, "preferred_roles": pref_roles})


def _answer_scale(pipeline) -> dict:
    pano = pipeline.panorama()
    nodes = pano.get("nodes", [])
    edges = pano.get("edges", [])
    roles = [n for n in nodes if n.get("type") == "Role"]
    skills = [n for n in nodes if n.get("type") == "Skill"]
    industries = sorted({n.get("industry") for n in roles if n.get("industry")})
    n_resume = len(getattr(pipeline, "_resumes", []) or [])
    text = (f"📊 当前图谱数据规模：\n"
            f"· 岗位节点：{len(roles)} 个\n"
            f"· 技能节点：{len(skills)} 个\n"
            f"· 关系连线：{len(edges)} 条\n"
            f"· 覆盖行业：{len(industries)} 个（{'、'.join(industries) or '—'}）\n"
            f"· 简历样本：{n_resume} 份")
    return _wrap("scale", text,
                 extra={"roles": len(roles), "skills": len(skills), "edges": len(edges)})


def _answer_metrics(config) -> dict:
    data_dir = Path(config.get("app", "data_dir", default="data"))
    report_path = data_dir / "evaluation_report.json"
    if not report_path.exists():
        return _wrap("metrics", "评测报告还未生成，请先运行 `python scripts/evaluate.py`。")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    jd = report.get("jd_parsing", {})
    resume_ex = report.get("resume_extraction", {})
    match = report.get("matching", {}) or report.get("resume_matching", {})
    lines = ["📈 系统评测指标（来自评测报告）："]
    if jd:
        lines.append(f"\n· JD 解析 F1：{jd.get('f1', 0):.1%}（P={jd.get('precision', 0):.1%} / R={jd.get('recall', 0):.1%}）")
    if resume_ex:
        lines.append(f"\n· 简历技能提取 F1：{resume_ex.get('f1', 0):.1%}")
    if match:
        lines.append(f"\n· 人岗匹配准确率：{match.get('accuracy', match.get('f1', 0)):.1%}")
    if "evidence_coverage" in report:
        lines.append(f"\n· 证据覆盖率：{report['evidence_coverage']:.1%}")
    if "hallucination_rate" in report:
        lines.append(f"\n· 幻觉率：{report['hallucination_rate']:.1%}（目标 ≤ 2%）")
    lines.append("\n\n完整报告可到「评测指标」页查看。")
    return _wrap("metrics", "\n".join(lines))


def _answer_new_roles(pipeline) -> dict:
    cands = pipeline.discover_new_roles()
    if not cands:
        return _wrap("new_roles", "当前时间窗口内暂未发现明显的新兴岗位候选簇。")
    lines = [f"💡 图谱发现了 {len(cands)} 个新兴岗位候选（按涌现分数排序）：\n"]
    for c in cands[:5]:
        score = c.get("emergence_score", 0)
        lines.append(f"· **{c.get('name', c.get('candidate', '?'))}**"
                     f"（涌现分 {score:.2f}，行业：{c.get('industry', '—')}）")
    lines.append("\n\n详情与证据可到「新岗位发现」页查看。")
    return _wrap("new_roles", "\n".join(lines),
                 extra={"candidates": [c.get("name", c.get("candidate", "")) for c in cands[:5]]})


def _answer_role_list(pipeline) -> dict:
    roles = pipeline.known_roles()
    lines = [f"📋 图谱覆盖 {len(roles)} 个岗位：", ""]
    for r in roles:
        lines.append(f"· {r}")
    lines.append("\n\n可以继续问我任意一个岗位需要什么技能～")
    return _wrap("role_list", "\n".join(lines), extra={"roles": roles})


def _answer_industries(pipeline) -> dict:
    pano = pipeline.panorama()
    roles = [n for n in pano.get("nodes", []) if n.get("type") == "Role"]
    by_industry: dict = {}
    for n in roles:
        by_industry.setdefault(n.get("industry") or "其他", []).append(n.get("label", ""))
    lines = ["🏭 图谱行业覆盖情况："]
    for ind, rs in sorted(by_industry.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"\n· **{ind}**（{len(rs)} 个岗位）：{'、'.join(rs[:4])}{'…' if len(rs) > 4 else ''}")
    return _wrap("industry", "\n".join(lines))


# ---------- 输出包装 ----------

_SUGGESTIONS = [
    "人工智能算法工程师需要什么技能？",
    "哪些岗位需要 Python？",
    "系统里有哪些岗位？",
    "图谱里有多少数据？",
    "评测指标怎么样？",
    "有什么新兴岗位？",
]


def _wrap(intent: str, answer_text: str, extra: Optional[dict] = None) -> dict:
    out = {"intent": intent, "answer": answer_text, "suggestions": _SUGGESTIONS}
    if extra:
        out.update(extra)
    return out
