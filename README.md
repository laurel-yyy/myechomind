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

## API surface

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Runtime, knowledge-base, Skills, and tool status |
| `POST` | `/chat` | Full memory → intent → RAG → multi-agent → persistence pipeline |
| `POST` | `/search` | Governed RAG search with rewrite and rerank |
| `POST` | `/knowledge/add` | Add one or more knowledge documents |
| `POST` | `/knowledge/upload` | Upload a UTF-8 `.txt`, `.md`, or `.json` knowledge document |
| `GET` | `/knowledge/stats` | Knowledge-base document count |
| `GET` / `POST` | `/skills`, `/skills/reload` | Inspect and hot-reload Skills |
| `GET` | `/monitor` | Component health, alerts, routing penalties, and tool counters |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/eval/run` | End-to-end evaluation and baseline regression checks |

## Docker Compose profiles

- Default services (no profile): `redis`, `chromadb`, `prometheus`
- `--profile with-nginx`: adds reverse-proxy nginx (port 80)
- `--profile with-app`: builds and runs the app container (port 8000)

Example — full stack via containers only:

```bash
docker compose --profile with-app --profile with-nginx up -d --build
```
