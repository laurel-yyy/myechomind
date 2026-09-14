"""FastAPI composition root and REST API for the MyEchoMind runtime."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from agents import AgentOrchestrator
from config import settings
from core.intents import get_spec
from core.skill_loader import SkillManager
from evaluation import DEFAULT_EVALUATION_CASES, EvaluationCase, Evaluator
from mcp.knowledge_base import KnowledgeBase
from mcp.knowledge_search_tool import KnowledgeSearchTool
from mcp.tool_manager import ToolManager
from memory.conversation_memory import ConversationMemory
from monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

CHAT_REQUESTS = Counter("myechomind_chat_requests_total", "Completed chat requests", ["intent", "agent"])
CHAT_LATENCY = Histogram("myechomind_chat_latency_seconds", "End-to-end chat latency")


@dataclass
class Runtime:
    """Shared application services, constructed once during lifespan startup."""

    memory: ConversationMemory
    knowledge_base: KnowledgeBase
    tools: ToolManager
    skills: SkillManager
    monitor: PerformanceMonitor
    orchestrator: AgentOrchestrator
    evaluator: Evaluator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=200)
    conversation_id: str | None = Field(default=None, max_length=200)
    use_rag: bool = True


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)
    rewrite_n: int = Field(default=3, ge=1, le=5)
    rerank: bool = True


class KnowledgeDocument(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str | None = Field(default=None, max_length=200)


class KnowledgeBatchRequest(BaseModel):
    documents: list[KnowledgeDocument] = Field(min_length=1, max_length=100)


class EvalCaseRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=10_000)
    expected_intent: str = Field(min_length=1, max_length=100)
    expected_group: str = Field(min_length=1, max_length=100)
    reference_answer: str | None = Field(default=None, max_length=20_000)


class EvalRequest(BaseModel):
    cases: list[EvalCaseRequest] | None = Field(default=None, max_length=100)


def _build_runtime() -> Runtime:
    monitor = PerformanceMonitor()
    skills = SkillManager()
    memory = ConversationMemory()
    knowledge_base = KnowledgeBase()
    tools = ToolManager([KnowledgeSearchTool(knowledge_base)], monitor=monitor)
    orchestrator = AgentOrchestrator(skill_manager=skills, monitor=monitor)
    evaluator = Evaluator(orchestrator=orchestrator)
    return Runtime(
        memory=memory,
        knowledge_base=knowledge_base,
        tools=tools,
        skills=skills,
        monitor=monitor,
        orchestrator=orchestrator,
        evaluator=evaluator,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.runtime = _build_runtime()
    logger.info("MyEchoMind runtime initialized")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="MyEchoMind API",
        version="0.1.0",
        description="Enterprise multi-agent support runtime with RAG, memory, Skills, monitoring, and evaluation.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        return {
            "status": "ok",
            "llm_mode": settings.llm_mode,
            "embedding_mode": settings.embedding_mode,
            "knowledge_base": runtime.knowledge_base.stats(),
            "skills": len(runtime.skills.list_skills()),
            "tools": runtime.tools.stats(),
        }

    @app.post("/chat")
    def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        conversation_id = payload.conversation_id or uuid.uuid4().hex
        started = time.perf_counter()
        memory_context = runtime.memory.build_context(
            payload.user_id, conversation_id, payload.message
        )
        intent, routing = runtime.orchestrator.route(payload.message, str(memory_context))
        docs: list[dict[str, Any]] = []
        rag_meta: dict[str, Any] = {"enabled": False, "reason": "intent_does_not_require_retrieval"}
        if payload.use_rag and _requires_rag(intent.intent):
            tool_result = runtime.tools.execute(
                "knowledge_search", {"query": payload.message, "top_k": settings.rag_topk}
            )
            docs = list((tool_result.data or {}).get("docs", []))
            rag_meta = {"enabled": True, "tool": tool_result.to_dict()}

        result = runtime.orchestrator.handle(
            payload.message,
            memory_context=memory_context,
            knowledge_docs=docs,
            intent_result=intent,
            routing_decision=routing,
        )
        runtime.memory.append_message(payload.user_id, conversation_id, "user", payload.message)
        runtime.memory.append_message(payload.user_id, conversation_id, "assistant", result.answer)
        runtime.memory.add_episodic(
            payload.user_id,
            conversation_id,
            f"User: {payload.message}\nAssistant: {result.answer}",
            metadata={"intent": result.intent.intent, "agent": result.routing.primary_agent},
        )
        elapsed = time.perf_counter() - started
        CHAT_REQUESTS.labels(result.intent.intent, result.routing.primary_agent).inc()
        CHAT_LATENCY.observe(elapsed)
        return {
            "conversation_id": conversation_id,
            "user_id": payload.user_id,
            "answer": result.answer,
            "intent": result.intent.to_dict(),
            "routing": result.routing.to_dict(),
            "knowledge": rag_meta,
            "latency_ms": round(elapsed * 1000, 2),
        }

    @app.post("/search")
    def search(payload: SearchRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        result = runtime.tools.execute(
            "knowledge_search",
            {
                "query": payload.query,
                "top_k": payload.top_k,
                "rewrite_n": payload.rewrite_n,
                "rerank": payload.rerank,
            },
        )
        return result.to_dict()

    @app.post("/knowledge/add")
    def add_knowledge(payload: KnowledgeBatchRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        ids = runtime.knowledge_base.add_batch(
            [
                {"id": doc.doc_id, "text": doc.text, "metadata": doc.metadata}
                for doc in payload.documents
            ]
        )
        return {"ids": ids, "count": len(ids)}

    @app.post("/knowledge/upload")
    async def upload_knowledge(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        runtime = _runtime(request)
        raw = await file.read()
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".txt", ".md", ".json"}:
            raise HTTPException(status_code=415, detail="Only .txt, .md, and .json uploads are supported")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Uploaded file must be UTF-8 encoded") from exc
        if suffix == ".json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="Invalid JSON document") from exc
        if not text.strip():
            raise HTTPException(status_code=400, detail="Uploaded document is empty")
        doc_id = runtime.knowledge_base.add(text, metadata={"source": file.filename or "upload"})
        return {"id": doc_id, "source": file.filename, "bytes": len(raw)}

    @app.get("/knowledge/stats")
    def knowledge_stats(request: Request) -> dict[str, Any]:
        return _runtime(request).knowledge_base.stats()

    @app.get("/skills")
    def list_skills(request: Request) -> dict[str, Any]:
        skills = _runtime(request).skills.list_skills()
        return {"count": len(skills), "skills": skills}

    @app.post("/skills/reload")
    def reload_skills(request: Request) -> dict[str, Any]:
        skills = _runtime(request).skills.reload()
        return {"count": len(skills), "skills": _runtime(request).skills.list_skills()}

    @app.get("/monitor")
    def monitor(request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        return {"monitor": runtime.monitor.summary(), "tools": runtime.tools.stats()}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/eval/run")
    def run_evaluation(payload: EvalRequest, request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        cases = (
            [
                EvaluationCase(
                    case_id=item.case_id,
                    message=item.message,
                    expected_intent=item.expected_intent,
                    expected_group=item.expected_group,
                    reference_answer=item.reference_answer,
                )
                for item in payload.cases
            ]
            if payload.cases is not None
            else DEFAULT_EVALUATION_CASES
        )
        return runtime.evaluator.run(cases).to_dict()

    return app


def _runtime(request: Request) -> Runtime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise RuntimeError("Application runtime has not been initialized")
    return runtime


def _requires_rag(intent: str) -> bool:
    """Avoid RAG cost for social, closure, and explicit handoff messages."""
    return intent not in {"greeting", "farewell", "feedback", "unknown", "human_handoff"}


app = create_app()
