# MyEchoMind

An enterprise-grade multi-agent customer-support / operations platform.

**Capability formula:** Intent Recognition + RAG + Memory + Multi-Agent Routing + Skills + Monitor + Evaluation

## What this project is

MyEchoMind receives a natural-language user request and turns it into a
governed pipeline: recognize the intent, retrieve relevant knowledge, route
to one or more specialist agents, generate a grounded reply, persist the
turn, and observe the whole thing so routing and evaluation can improve over
time. It is designed as a *runtime*, not a chatbot demo — every step is
independently testable, swappable, and observable.

Concretely, one `POST /chat` walks the following path:

```
request
  │
  ▼
Memory       Redis working memory + ChromaDB episodic / user profile
  │
  ▼
Intent       LLM (0.7) + embedding cosine (0.2) + keyword (0.1) fusion
             18 fine-grained intents, 4 groups, urgency, entity extraction
  │
  ▼
RAG          Query rewrite → parallel recall → LLM rerank → top-k
             (skipped for greeting / farewell / handoff / unknown)
  │
  ▼
Orchestrator Primary + supporting agents, routing reason, confidence
             Monitor penalties feed back into the routing score
  │
  ▼
Skills       Hot-reloadable SKILL.md files inject scoped rules per agent
  │
  ▼
Agent(s)     BaseAgent + General / Technical / Billing specialists
             Graceful degradation when the LLM is unreachable
  │
  ▼
Persist      Working memory append + episodic memory upsert
  │
  ▼
Monitor      Latency, success rate, breaker state, anomaly detection,
             routing penalty updates
  │
  ▼
response
```

Every component sits behind an abstraction (LLM, embedding, Redis, Chroma)
so the same code runs in a **mock** mode (deterministic, no API keys, no
network) or a **real** mode (Anthropic / DeepSeek + sentence-transformers +
containerized Redis / ChromaDB). Mode selection is a single `.env` flip.

## Repository layout

```
config/       Centralized Pydantic Settings + prometheus / nginx configs
core/         LLM & embedding abstractions, intent taxonomy,
              three-path fusion recognizer, dynamic Skills loader
memory/       Redis working memory + summary,
              ChromaDB episodic memory + user profile
mcp/          RAG knowledge base + tool governance harness
              (query rewrite, LLM rerank, TTL cache, circuit breaker,
              timeout, fallback)
agents/       BaseAgent + General / Technical / Billing specialists,
              AgentOrchestrator with monitor-penalty-aware routing
monitor/      Online performance monitor: rolling latency, success rate,
              z-score anomaly, breaker state, routing penalty feedback
evaluation/   LLM-as-Judge end-to-end evaluator with baseline regression
skills/       Hot-reloadable business rules per agent (SKILL.md)
api/          FastAPI composition root + REST surface
scripts/      Per-module smoke tests + end-to-end integration smoke
frontend/     Vue 3 + Vite console (single-file SPA, no UI framework)
```

## Quick start

### 0. Prerequisites

- Python 3.11+
- Docker Desktop (for Redis and ChromaDB containers)
- Node.js 20+ (only if you want the console frontend)

### 1. Bring up infrastructure

```bash
docker compose up -d redis chromadb prometheus
docker compose ps
```

You should see `myechomind-redis`, `myechomind-chromadb`, and
`myechomind-prometheus` in `Up` state.

### 2. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # macOS/Linux

pip install -r requirements.txt
copy .env.example .env                 # Windows
# cp .env.example .env                 # macOS/Linux
```

The default `.env` starts everything in **mock** mode, so nothing else is
required to boot the server.

### 3. Start the backend

```bash
uvicorn api.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Health:     http://localhost:8000/health
- Prometheus: http://localhost:9090

### 4. (Optional) Start the frontend console

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to the
backend on port 8000, so no CORS setup is needed. The console has six
panels: Chat, Knowledge, Search, Skills, Monitor, Eval.

### 5. Switch to real LLM (optional)

Edit `.env`:

```
LLM_MODE=real
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
# Only if your key is org-scoped rather than workspace-scoped:
# ANTHROPIC_WORKSPACE_ID=wrkspc-...
```

DeepSeek works out of the box via the same client — just point the base URL
and model at their Anthropic-compatible endpoint:

```
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=deepseek-chat
```

Restart uvicorn. The `/health` endpoint reports the active mode; the
console header shows it as a chip.

### 6. Verify everything works

```bash
# per-module smoke tests
.venv\Scripts\python.exe scripts\smoke_m0.py   # config + LLM abstraction
.venv\Scripts\python.exe scripts\smoke_m1.py   # memory (Redis + ChromaDB)
.venv\Scripts\python.exe scripts\smoke_m2.py   # intent recognizer
.venv\Scripts\python.exe scripts\smoke_m3.py   # RAG + tool governance

# end-to-end smoke against a running server on :8000
.venv\Scripts\python.exe scripts\smoke_e2e.py
```

The E2E smoke reads `/health` to detect mock vs real mode and adapts its
assertions automatically.

## Runtime modes

Every knob is env-driven; no business code changes when you swap modes.

| Component  | Lightweight default   | Real                                         |
|------------|-----------------------|----------------------------------------------|
| LLM        | Rule-based mock       | Anthropic / DeepSeek (Anthropic-compatible)  |
| Embedding  | Hash pseudo-vector    | sentence-transformers (`all-MiniLM-L6-v2`)   |
| Redis      | Docker Redis          | Same, with password / cluster if needed      |
| ChromaDB   | Docker HTTP server    | Persistent local client (needs MSVC toolchain on Windows) |

## API surface

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/health` | Runtime, knowledge-base, Skills, and tool status |
| `POST` | `/chat` | Full memory → intent → RAG → multi-agent → persistence pipeline |
| `POST` | `/search` | Governed RAG search with rewrite and rerank |
| `POST` | `/knowledge/add` | Add one or more knowledge documents |
| `POST` | `/knowledge/upload` | Upload a UTF-8 `.txt`, `.md`, or `.json` document (≤10 MB) |
| `GET`  | `/knowledge/stats` | Knowledge-base document count |
| `GET`  | `/skills` | List loaded Skills with agents and keywords |
| `POST` | `/skills/reload` | Re-scan the `skills/` directory from disk |
| `GET`  | `/monitor` | Component health, alerts, routing penalties, tool counters |
| `GET`  | `/metrics` | Prometheus metrics endpoint |
| `POST` | `/eval/run` | End-to-end evaluation and baseline regression check |

## Docker Compose profiles

- Default services (no profile): `redis`, `chromadb`, `prometheus`
- `--profile with-nginx`: adds reverse-proxy nginx on port 80
- `--profile with-app`: builds and runs the app container on port 8000

Full stack via containers only:

```bash
docker compose --profile with-app --profile with-nginx up -d --build
```

## Troubleshooting

**`chromadb` install fails on Windows.** The default requirements pin
`chromadb-client` (pure Python, HTTP only). Keep `CHROMA_MODE=http` and let
the Docker ChromaDB service do the work. Installing the full `chromadb`
package for `CHROMA_MODE=local` needs MSVC Build Tools.

**Real-mode `/chat` returns a "temporarily unavailable" reply.** The
upstream LLM raised. Check `logs/` for the exact error. Common causes:
retired model ID (update `ANTHROPIC_MODEL`), org-scoped API key (set
`ANTHROPIC_WORKSPACE_ID` or generate a workspace-scoped key), rate limit.
The pipeline is intentionally resilient: intent detection, retrieval,
routing, memory, and monitoring keep working while `chat` degrades.

**`/monitor` shows `high_latency` alerts in real mode.** Expected — the
default threshold is 3000 ms and real Claude responses often exceed that.
Adjust `MONITOR_LATENCY_MS_THRESHOLD` in `.env` for your workload.

**High latency on `/eval/run`.** Real-mode eval drives 8 default cases
through roughly three LLM roundtrips each. A full run typically takes 2–4
minutes.
