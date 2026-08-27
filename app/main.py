"""FastAPI 应用：岗位能力图谱系统 API（对齐技术路线文档 API 契约）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
# StaticFiles removed: replaced by FileResponse route to fix Windows content-length bug
from pydantic import BaseModel, Field

from .config import ROOT_DIR, load_config
from .skills import SKILL_DICT
from .pipeline import Pipeline, bootstrap

config = load_config()
pipeline: Optional[Pipeline] = None


class AsciiSafeJSONMiddleware:
    """ASGI 中间件：统一修正所有响应的 Content-Length，并对 JSON 做 ASCII 化。

    针对特定 fastapi/starlette 版本组合下，含中文的响应（JSON 或 HTML）
    的 Content-Length 按「字符数」而非「UTF-8 字节数」计算，导致
    "Response content longer than Content-Length" 错误。
    本中间件收集全部响应体字节，按实际字节数重写 content-length 头，
    对 application/json 额外用 ensure_ascii=True 重编码，双保险。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = {"start": None, "status": 200, "ctype": "", "chunks": []}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                state["start"] = message
                state["status"] = message.get("status", 200)
                headers = {
                    k.decode("latin-1").lower(): v.decode("latin-1")
                    for k, v in message.get("headers", [])
                }
                state["ctype"] = headers.get("content-type", "")
            elif message["type"] == "http.response.body":
                state["chunks"].append(message.get("body", b""))

        await self.app(scope, receive, send_wrapper)

        if state["start"] is None:
            return

        # 无体（204/304）或没有任何 body 消息：原样转发
        if state["status"] in (204, 304) or not state["chunks"]:
            await send(state["start"])
            return

        body = b"".join(state["chunks"])

        # JSON：额外 ASCII 化，保证字符数==字节数（双保险）
        if "application/json" in state["ctype"]:
            try:
                obj = json.loads(body.decode("utf-8"))
                body = json.dumps(
                    obj, ensure_ascii=True, allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except Exception:
                pass

        # 按实际字节数重写 content-length
        headers = [
            (k, v) for k, v in state["start"].get("headers", [])
            if k.lower() != b"content-length"
        ]
        headers.append((b"content-length", str(len(body)).encode("latin-1")))
        state["start"]["headers"] = headers
        await send(state["start"])
        await send({"type": "http.response.body", "body": body})


class SafeJSONResponse(JSONResponse):
    """ASCII 安全的 JSON 响应。

    强制 ensure_ascii=True，把非 ASCII 字符转成 \\uXXXX 纯 ASCII，
    规避部分 fastapi/starlette/uvicorn 版本组合下，中文 JSON 响应
    Content-Length 按「字符数」而非「字节数」计算导致的
    "Response content longer than Content-Length" 错误。
    前端 fetch().json() 解析时会自动还原中文，界面无感知。
    """

    def render(self, content) -> bytes:
        return json.dumps(
            content, ensure_ascii=True, allow_nan=False,
            indent=None, separators=(",", ":"),
        ).encode("utf-8")


# 兜底补丁：直接把 JSONResponse 基类的 render 也替换成 ASCII 安全版本。
# 无论 FastAPI 版本是否尊重 default_response_class，所有 JSON 响应都生效，
# 从根上规避「中文 JSON 的 Content-Length 按字符数算、按字节数发」导致
# 的 "Response content longer than Content-Length" 错误。
JSONResponse.render = SafeJSONResponse.render


app = FastAPI(
    title="岗位能力图谱系统",
    version="0.1.0",
    description="多源异构数据驱动岗位和能力图谱构建与动态演化分析系统",
    default_response_class=SafeJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_middleware(AsciiSafeJSONMiddleware)

# 静态文件：用 FileResponse 手动路由，绕过 Starlette StaticFiles 在 Windows 上
# content-length: 0 的 bug。FileResponse 直接读文件并设置正确的 content-length。
@app.get("/static/{file_path:path}")
def serve_static(file_path: str):
    web_dir = ROOT_DIR / "web"
    full_path = web_dir / file_path
    # 安全检查：防止路径穿越
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(web_dir.resolve())):
            raise HTTPException(status_code=403, detail="Forbidden")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid path")
    if full_path.exists() and full_path.is_file():
        return FileResponse(str(full_path))
    raise HTTPException(status_code=404, detail="File not found")


def get_pipeline() -> Pipeline:
    global pipeline
    if pipeline is None:
        data_dir = Path(config.get("app", "data_dir", default="data"))
        pipeline = bootstrap(data_dir)
    return pipeline


# ---------- 请求模型 ----------
class DiscoverRequest(BaseModel):
    known_roles: list[str] = Field(default_factory=list)


class TimelineRequest(BaseModel):
    role_name: str
    window_a: str = "2020-01-01"
    window_b: str = "2026-01-01"


class ResumeParseRequest(BaseModel):
    text: str = ""
    file_path: str = ""
    resume_id: str = ""
    name: str = ""


class MatchRequest(BaseModel):
    resume_text: str
    role_name: str
    resume_id: str = ""
    name: str = ""


class LoginRequest(BaseModel):
    username: str = ""
    password: str = ""


class IngestRequest(BaseModel):
    jobs: list[dict] = Field(default_factory=list)


# ---------- 认证（演示用，默认账号 admin/admin123） ----------
@app.post("/api/v1/auth/login")
def auth_login(req: LoginRequest):
    if req.username == "admin" and req.password == "admin123":
        return {"token": "demo-token-" + req.username,
                "user": {"username": req.username, "display_name": "管理员",
                         "role": "admin", "avatar": ""}}
    raise HTTPException(status_code=401, detail="用户名或密码错误")


# ---------- 健康检查 ----------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "job-ability-graph", "version": "0.1.0",
            "render": "ascii-safe", "middleware": "ascii-safe-json"}


@app.get("/api/v1/health")
def health_v1() -> dict:
    return {"status": "ok"}


# ---------- 数据接入 ----------
@app.post("/api/v1/ingestion/jobs")
def create_ingestion_job(req: IngestRequest):
    """创建数据导入任务（简化：直接治理并返回质量报告）。"""
    p = get_pipeline()
    if req.jobs:
        p._raw_jobs = req.jobs
    jobs = p.run_ingestion()
    return {"job_count": len(jobs), "jobs": jobs}


@app.get("/api/v1/ingestion/jobs/{job_id}")
def get_ingestion_job(job_id: str):
    p = get_pipeline()
    for j in p._jobs:
        if j.get("job_id") == job_id:
            return j
    raise HTTPException(status_code=404, detail="job not found")


# ---------- 新岗位发现 ----------
@app.post("/api/v1/roles/discover")
def discover_roles_endpoint(req: DiscoverRequest):
    p = get_pipeline()
    roles = p.discover_new_roles(req.known_roles)
    return {"candidates": roles, "count": len(roles)}


@app.get("/api/v1/roles")
def list_roles():
    p = get_pipeline()
    return {"roles": p.known_roles()}


# ---------- 岗位演化时间线 ----------
@app.get("/api/v1/roles/{role_name}/timeline")
def role_timeline(role_name: str, window_a: str = "2020-01-01", window_b: str = "2026-01-01"):
    p = get_pipeline()
    return p.role_timeline(role_name, window_a, window_b)


# ---------- 技能本体 ----------
@app.get("/api/v1/skills/ontology")
def skills_ontology():
    """技能本体库：技能名 + 分类 + 别名（供知识库页使用）。"""
    skills = [
        {"name": name, "category": meta.get("category", "其他"),
         "aliases": meta.get("aliases", [])}
        for name, meta in SKILL_DICT.items()
    ]
    return {"skills": skills, "count": len(skills)}


# ---------- 全景图谱 ----------
@app.get("/api/v1/graph/panorama")
def panorama(industry: str = "", tech_stack: str = "", level: str = ""):
    p = get_pipeline()
    return p.panorama(industry, tech_stack, level)


@app.get("/api/v1/graph/versions")
def graph_versions():
    p = get_pipeline()
    return {"versions": p.graph.list_versions()}


# ---------- 简历解析 ----------
@app.post("/api/v1/resumes/parse")
def parse_resume(req: ResumeParseRequest):
    p = get_pipeline()
    from .resume_match import parse_resume as _parse
    resume = _parse(req.text or "", req.resume_id or "resume-1", req.name)
    return resume.to_dict()


@app.post("/api/v1/resumes/upload")
async def upload_resume(file: UploadFile = File(...)):
    """上传 PDF/Word 简历，解析为文本。"""
    content = await file.read()
    text = _extract_resume_text(content, file.filename or "")
    from .resume_match import parse_resume as _parse
    resume = _parse(text, file.filename or "upload", "")
    return resume.to_dict()


def _extract_resume_text(content: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(content))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF 解析失败: {e}")
    if name.endswith((".docx", ".doc")):
        try:
            import io
            from docx import Document
            doc = Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Word 解析失败: {e}")
    # 纯文本
    return content.decode("utf-8", errors="ignore")


# ---------- 人岗匹配 ----------
@app.post("/api/v1/matching/diagnose")
def match_diagnose(req: MatchRequest):
    p = get_pipeline()
    return p.parse_and_match(req.resume_text, req.role_name, req.resume_id, req.name)


# ---------- 学习路径 ----------
@app.post("/api/v1/learning-paths")
def learning_paths(req: MatchRequest):
    p = get_pipeline()
    result = p.parse_and_match(req.resume_text, req.role_name, req.resume_id, req.name)
    return {"role_name": req.role_name, "learning_path": result["learning_path"],
            "missing_skills": result["match"]["missing_skills"]}


# ---------- 审核 ----------
@app.post("/api/v1/reviews/{claim_id}/approve")
def approve_claim(claim_id: str):
    return {"claim_id": claim_id, "status": "approved", "note": "审核通过（演示）"}


# ---------- 评测 ----------
@app.get("/api/v1/metrics/evaluation")
def metrics_evaluation():
    report_path = Path(config.get("app", "data_dir", default="data")) / "evaluation_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {"message": "请先运行 scripts/evaluate.py 生成评测报告"}


# ---------- 前端 ----------
@app.get("/")
def index():
    web = ROOT_DIR / "web" / "index.html"
    if web.exists():
        return FileResponse(str(web))
    return JSONResponse({"message": "web/index.html 不存在，请先构建前端"})


@app.get("/favicon.ico")
def favicon():
    return JSONResponse({}, status_code=204)
