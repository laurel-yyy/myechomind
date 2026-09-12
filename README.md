# MyEchoMind

An enterprise-grade multi-agent customer-support / operations platform.

**Capability formula**: Intent Recognition + RAG + Memory + Multi-Agent Routing + Skills + Monitor + Evaluation

## Quick start

### 1. Bring up infrastructure (Redis + ChromaDB + Prometheus)

```bash
docker compose up -d redis chromadb prometheus
```

### 2. Install dependencies and run the app

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
copy .env.example .env           # Windows
# cp .env.example .env           # macOS/Linux

uvicorn api.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Prometheus: http://localhost:9090

## Layout

```
config/       Centralized Pydantic Settings + prometheus / nginx configs
core/         LLM / Embedding abstractions, intent recognition, skills loader
memory/       Redis working memory + ChromaDB episodic / user profile
mcp/          RAG knowledge base + tool governance (rewrite / rerank / breaker / cache)
agents/       BaseAgent + 3 specialized agents + AgentOrchestrator
monitor/      Online monitoring + monitor_penalty feedback into routing
evaluation/   LLM-as-Judge end-to-end evaluator
skills/       Hot-reloadable business rules (SKILL.md)
api/          FastAPI entry point
```

## Runtime modes

Toggle via `.env`; no business code changes required.

| Component  | Lightweight default | Real                       |
|------------|---------------------|----------------------------|
| LLM        | Rule-based mock     | Anthropic / DeepSeek       |
| Embedding  | Hash pseudo-vector  | sentence-transformers      |
| Redis      | fakeredis (in-mem)  | Real Redis                 |
| ChromaDB   | Local Persistent    | HTTP server                |

## Docker Compose profiles

- Default services (no profile): `redis`, `chromadb`, `prometheus`
- `--profile with-nginx`: adds reverse-proxy nginx (port 80)
- `--profile with-app`: builds and runs the app container (port 8000)

Example — full stack via containers only:

```bash
docker compose --profile with-app --profile with-nginx up -d --build
```
