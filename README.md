# LLM Observability with Elastic APM & OpenLLMetry

Full-stack LLM monitoring using open standards. Track cost, latency, tokens, and errors across OpenAI and Anthropic — all in Kibana.

![Dashboard](docs/img/screenshot-dashboard.png)

## Features

- **Zero-code LLM instrumentation** — OpenLLMetry auto-captures model, tokens, prompts, and response metadata
- **Automatic cost tracking** — Custom span exporter calculates dollar cost per call using [LiteLLM's pricing database](https://github.com/BerriAI/litellm) (1,500+ models)
- **8-panel Kibana dashboard** — Cost distribution, token usage, latency (avg + P95), error rates
- **Multi-agent trace correlation** — Nested workflows with parallel execution across GPT and Claude
- **One-command setup** — Docker Compose brings up the full stack in 60 seconds

## Architecture

```
Flask App → OpenLLMetry SDK → OTel Collector → APM Server → Elasticsearch → Kibana
```

| Service | Image | Port |
|---|---|---|
| Elasticsearch | `elastic/elasticsearch:8.11.0` | [localhost:9200](http://localhost:9200) |
| Kibana | `elastic/kibana:8.11.0` | [localhost:5601](http://localhost:5601) |
| OTel Collector | `otel/opentelemetry-collector-contrib:0.91.0` | 4318 (HTTP), 4317 (gRPC) |
| APM Server | `elastic/apm-server:8.11.0` | 8200 |
| Flask App | Local build | [localhost:5001](http://localhost:5001) |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/maheshbabugorantla/llm-observability-with-elasticapm.git
cd llm-observability-elastic
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2. Start the stack

```bash
docker compose up --build -d
```

Wait ~60–90 seconds for all services. To check readiness:

```bash
./scripts/start.sh
```

Or verify manually:

```bash
curl -s -u elastic:changeme http://localhost:9200 | head -5   # Elasticsearch
curl -s http://localhost:5601/api/status | head -1              # Kibana
curl -s http://localhost:5001/health                            # Flask app
```

### 3. Generate traces

```bash
# Single recipe
curl -X POST http://localhost:5001/recipe/generate \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "dish_name": "Spaghetti Carbonara", "cuisine_type": "Italian", "servings": 4}'

# Multi-agent workflow (4 AI agents designing a restaurant menu)
curl -X POST http://localhost:5001/menu/design \
  -H "Content-Type: application/json" \
  -d '{"cuisine": "Japanese", "menu_type": "omakase", "courses": 3, "dietary_requirements": ["pescatarian"], "budget": "premium", "season": "winter", "occasion": "business_dinner"}'
```

Or run the test suites:

```bash
./scripts/test-api.sh               # 5 basic API tests
./scripts/test-multiagent-menu.sh   # Multi-agent workflow (3 scenarios)
```

> **Cost note:** A full test run could cost roughly $0.50–$2.00 in API calls.

### 4. Import the dashboard

```bash
./kibana/setup-dashboards.sh
```

### 5. View in Kibana

Open [http://localhost:5601](http://localhost:5601) (login: `elastic` / `changeme`)

**APM traces:** Observability → APM → Services → recipe-generator-service

![APM Service Overview](docs/img/screenshot-apm-service-overview.png)

**Trace waterfall:** Click any transaction to see the full span tree with nested agent calls:

![Trace Waterfall](docs/img/screenshot-trace-waterfall.png)

**Span metadata:** Click any `openai.chat` or `anthropic.chat` span to see captured gen_ai attributes, token counts, and injected cost data:

![Span Metadata](docs/img/screenshot-span-metadata.png)
![Cost Data](docs/img/screenshot-span-cost-data.png)

**Dashboard:** [http://localhost:5601/app/dashboards#/view/llm-observability-dashboard](http://localhost:5601/app/dashboards#/view/llm-observability-dashboard)

> If the dashboard shows "No results found", widen the time range to "Last 30 days" using the date picker.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/recipe/generate` | Generate a recipe with one provider |
| `POST` | `/recipe/compare` | Generate with both OpenAI and Claude |
| `POST` | `/recipe/batch` | Generate multiple recipes |
| `POST` | `/menu/design` | Multi-agent restaurant menu design |

---

## Captured Attributes

Every LLM span automatically includes:

| Field | Example | Storage in ES |
|---|---|---|
| `gen_ai.system` | `openai` | `labels.gen_ai_system` |
| `gen_ai.request.model` | `gpt-4o` | `labels.gen_ai_request_model` |
| `gen_ai.usage.input_tokens` | `165` | `numeric_labels.gen_ai_usage_input_tokens` |
| `gen_ai.usage.output_tokens` | `1500` | `numeric_labels.gen_ai_usage_output_tokens` |
| `gen_ai.cost.total_usd` | `0.003041` | `numeric_labels.gen_ai_cost_total_usd` |
| `gen_ai.cost.input_usd` | `0.000041` | `numeric_labels.gen_ai_cost_input_usd` |
| `gen_ai.cost.output_usd` | `0.003` | `numeric_labels.gen_ai_cost_output_usd` |
| `gen_ai.prompt.0.content` | *(system prompt)* | `labels.gen_ai_prompt_0_content` |

> Dots become underscores in Elasticsearch. Strings land in `labels.*`, numbers in `numeric_labels.*`.

---

## Cost Tracking

OpenLLMetry captures tokens but not dollar cost. The `CostEnrichingSpanExporter` in `app/llm_cost_injector.py` wraps the OTLP exporter to:

1. Intercept spans with `gen_ai.system` attribute
2. Look up per-token pricing from LiteLLM (1,500+ models, synced from GitHub)
3. Calculate and inject `gen_ai.cost.{input_usd, output_usd, total_usd}`
4. Forward the enriched span to the real exporter

Non-LLM spans pass through untouched. Pricing is cached locally and auto-refreshes every 24 hours.

```python
from llm_cost_injector import inject_llm_cost_tracking

Traceloop.init(app_name="recipe-generator-service", ...)
inject_llm_cost_tracking()  # wraps the active exporter
```

---

## Project Structure

```
├── app/
│   ├── app.py                    # Flask app with @workflow / @task decorators
│   ├── llm_cost_injector.py      # CostEnrichingSpanExporter + LiteLLM pricing
│   ├── Dockerfile
│   └── requirements.txt
├── kibana/
│   ├── llm-observability-dashboards.ndjson
│   └── setup-dashboards.sh
├── scripts/
│   ├── start.sh                  # Start stack + health checks
│   ├── test-api.sh               # 5 basic API tests
│   └── test-multiagent-menu.sh   # Multi-agent workflow tests
├── docker-compose.yml
├── otel-collector-config.yaml
├── .env.example
└── LICENSE (MIT)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| ES won't start (macOS) | Docker Desktop → Settings → Resources → 8GB+ RAM |
| No traces in Kibana | Check `curl http://localhost:5001/health` and `docker compose logs otel-collector` |
| Flask 500 errors | Verify API keys in `.env`, then `docker compose restart flask-app` |
| Dashboard empty | Widen time range to "Last 30 days" in date picker |
| Kibana login | Username: `elastic`, Password: `changeme` |
| Service Map → "Start trial" | Requires Platinum license (free 30-day trial available) |
| Port conflicts | Stop conflicting services or edit ports in `docker-compose.yml` |

## Stopping

```bash
docker compose down      # Stop containers, keep data
docker compose down -v   # Stop and delete all data
```

---

## License

MIT — see [LICENSE](LICENSE).
