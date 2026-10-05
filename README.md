# NEXUS AI

**Navigated Execution & eXam Unified System**

A career-path auditor for B.Tech students. It inventories your current skills,
evaluates feasibility across five career paths, eliminates goals that compete
for the same hours, and returns a tiered roadmap.

Built with FastAPI, LangChain tool-calling agents, and Pydantic v2.

---

## The problem it solves

A student decides to "prepare for placements, GATE, CAT, and an abroad MS"
simultaneously. These goals conflict: different syllabi, different question
styles, different deadlines — all drawing on the same weekly hours. NEXUS
quantifies that conflict instead of letting the student discover it by
burning out.

## How it works

Three LangChain tools do all arithmetic over a curated knowledge base:

| Tool | Purpose |
|---|---|
| `audit_skill_inventory` | Maps self-declared skills to syllabus topics, then computes gap hours, buffer, and a feasibility flag for one path |
| `query_syllabus_knowledge_base` | Returns a Basic → Intermediate → Advanced blueprint for one topic |
| `calculate_path_elimination_matrix` | Ranks all target paths, quantifies conflict overhead, and produces the KEEP/DROP verdict |

The LLM **narrates** tool output. It never computes coverage, hours, or
feasibility itself — those are deterministic. The system prompt forbids
inventing numbers.

### Feasibility flags

- `HIGH` — required months ≤ 70% of available runway
- `MODERATE` — required months ≤ 100% of runway
- `CRITICAL_TIMELINE` — everything else

Arithmetic only. No opinions.

### Supported paths

`PLACEMENTS` · `GATE_CS` · `GATE_DA` · `CAT` · `STUDY_ABROAD`

Aliases are accepted (`"gate cse"`, `"iim cat"`, `"masters abroad"`, …).
21 topics in the knowledge base, each with prerequisites and hour estimates.

---

## Quick start

```bash
pip install -r requirements.txt
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
python main.py
```

Open http://localhost:8000 for the web UI, or http://localhost:8000/docs for
the API.

### Works without any API key

With no key configured, NEXUS runs its **offline deterministic engine**:
`mode: "OFFLINE_DETERMINISTIC"`, `status: "partial"`. You still get the full
audit, elimination matrix, roadmap, and markdown report. Only the prose
narration is missing. This makes the app safe to demo and deploy with zero
credentials.

---

## Configuration

All configuration is environment variables. Copy `.env.example` and edit.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Credentials. Required for agent mode. |
| `OPENAI_BASE_URL` | OpenAI | Any OpenAI-compatible endpoint |
| `OPENAI_MODEL` | provider default | Override the model |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address |
| `RELOAD` / `WORKERS` / `LOG_LEVEL` | `false` / `1` / `info` | Uvicorn options |

### Using a different provider

NEXUS speaks the OpenAI protocol, so any compatible endpoint works with **no
code changes**. The default model follows the provider automatically.

```bash
# Google AI Studio (most generous free tier)
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/

# Groq
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1

# Ollama, fully local
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:14b

# GitHub Models / OpenRouter / Cerebras - also OpenAI-compatible
```

> Provider defaults exist because sending `gpt-4o-mini` to Groq fails with
> `model_not_found`. `GET /health` reports the resolved provider, model, and
> whether the model came from `OPENAI_MODEL` or the provider default.

> `Cerebras`'s free tier caps context near 8K, which an agent loop with three
> JSON-returning tools will exceed. Use it for text, not for NEXUS.

---

## Verifying your setup

```bash
curl -X POST http://localhost:8000/api/v1/llm-check
```

NEXUS is an agent, so a model that chats but cannot emit tool calls is useless
here — the most common failure when swapping providers. This preflight tests a
text round-trip **and** a tool-call round-trip, and returns a `remedy` string
naming the actual fix for `model_not_found`, quota, bad-key, and context-length
failures.

Want `ok: true` with both `text_roundtrip.ok` and `tool_calling.ok` true.

---

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI |
| `/health` | GET | Liveness, provider, circuit-breaker state |
| `/docs` | GET | OpenAPI docs |
| `/api/v1/diagnose` | POST | Full career diagnosis |
| `/api/v1/catalogue` | GET | The deterministic knowledge base |
| `/api/v1/llm-check` | POST | LLM preflight |

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Aarav",
    "known_skills": ["Python", "DSA", "SQL"],
    "target_paths": ["GATE_CS", "CAT"],
    "months_available": 6,
    "weekly_hours": 40
  }'
```

### Example response

```json
{
  "status": "partial",
  "mode": "OFFLINE_DETERMINISTIC",
  "primary_path": "GATE_CS",
  "feasibility": { "GATE_CS": "MODERATE", "CAT": "MODERATE" },
  "coverage": { "GATE_CS": { "weighted_coverage_percent": 25.3, "...": "" } },
  "elimination_matrix": { "elimination_matrix": [], "...": "" },
  "roadmap": [],
  "report": "# NEXUS Career Audit\n...",
  "warnings": ["LLM unavailable - used deterministic engine."]
}
```

### Tool errors are structured

Tools return JSON with `status: "ERROR"` and a machine-readable `error_code`,
plus `supported_paths`, `supported_topics`, and a `recovery_hint`. Nothing
raises, so the agent can self-correct instead of crashing.

```json
{
  "tool": "audit_skill_inventory",
  "status": "ERROR",
  "error_code": "UNMAPPED_PATH",
  "message": "'martian' is not a supported target path.",
  "supported_paths": ["PLACEMENTS", "GATE_CS", "GATE_DA", "CAT", "STUDY_ABROAD"],
  "recovery_hint": "Do not guess. Re-read the supported lists above..."
}
```

---

## Architecture notes

**The agent is built lazily** as a process-wide singleton, so importing the
module never requires credentials and `/health` answers without them.

**An LLM circuit breaker** stops retrying permanent failures (quota, auth).
Without it, every request would re-attempt a doomed call and burn its full
retry budget; the breaker cuts repeat failures to near-instant.

**Ranking is single-sourced** through one `rank_paths()` function, so the
endpoint, the matrix tool, and the offline engine cannot disagree about which
path is primary.

**Bandwidth uses gap hours ÷ runway**, not total syllabus hours. Using the full
syllabus produced a bogus 346% utilisation figure that looked like a hard
bottleneck where none existed.

---

## Project layout

```
main.py            Single-file application: knowledge base, tools, agent, API
key_manager.py     Optional multi-key rotation with health scoring
static/            Web UI (vanilla HTML/CSS/JS, no build step)
requirements.txt   Pinned dependencies
.env.example       Configuration template
```

---

## Requirements

Python 3.10+. A GPU is not required for the API, but a local Ollama model needs
roughly 16 GB RAM for a usable 14B model on CPU.

## License

MIT