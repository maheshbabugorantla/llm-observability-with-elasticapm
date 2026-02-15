#!/usr/bin/env bash
#
# setup-dashboards.sh
# Creates LLM Observability dashboards in Kibana via the Saved Objects API.
# Imports the NDJSON file and creates a data view for traces-apm*.
#

set -euo pipefail

KIBANA_URL="${KIBANA_URL:-http://localhost:5601}"
KIBANA_USER="${KIBANA_USER:-elastic}"
KIBANA_PASSWORD="${KIBANA_PASSWORD:-changeme}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NDJSON_FILE="${SCRIPT_DIR}/llm-observability-dashboards.ndjson"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -------------------------------------------------------------------
# 1. Wait for Kibana to be healthy
# -------------------------------------------------------------------
info "Waiting for Kibana at ${KIBANA_URL}..."

MAX_WAIT=90
ELAPSED=0
while true; do
  STATUS=$(curl -s -u "${KIBANA_USER}:${KIBANA_PASSWORD}" "${KIBANA_URL}/api/status" 2>/dev/null | grep -o '"level":"[^"]*"' | head -1 || true)
  if echo "$STATUS" | grep -q '"level":"available"'; then
    info "Kibana is available"
    break
  fi
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    error "Kibana did not become available within ${MAX_WAIT}s"
    exit 1
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  echo "  ...waited ${ELAPSED}s"
done

# -------------------------------------------------------------------
# 2. Create the data view (index pattern) for traces-apm*
# -------------------------------------------------------------------
info "Creating data view for traces-apm*..."

DATA_VIEW_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  "${KIBANA_URL}/api/data_views/data_view" \
  -H "kbn-xsrf: true" \
  -H "Content-Type: application/json" \
  -u "${KIBANA_USER}:${KIBANA_PASSWORD}" \
  -d '{
    "data_view": {
      "id": "traces-apm-data-view",
      "title": "traces-apm*",
      "timeFieldName": "@timestamp",
      "allowNoIndex": true
    },
    "override": true
  }')

HTTP_CODE=$(echo "$DATA_VIEW_RESPONSE" | tail -1)
if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
  info "Data view created successfully"
elif [ "$HTTP_CODE" -eq 409 ]; then
  warn "Data view already exists (409) — continuing"
else
  warn "Data view creation returned HTTP ${HTTP_CODE} — attempting to continue"
fi

# -------------------------------------------------------------------
# 3. Import saved objects (visualizations + dashboard)
# -------------------------------------------------------------------
if [ ! -f "$NDJSON_FILE" ]; then
  error "NDJSON file not found: ${NDJSON_FILE}"
  exit 1
fi

info "Importing saved objects from ${NDJSON_FILE}..."

IMPORT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  "${KIBANA_URL}/api/saved_objects/_import?overwrite=true" \
  -H "kbn-xsrf: true" \
  -u "${KIBANA_USER}:${KIBANA_PASSWORD}" \
  --form file=@"${NDJSON_FILE}")

HTTP_CODE=$(echo "$IMPORT_RESPONSE" | tail -1)
BODY=$(echo "$IMPORT_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -ge 200 ] && [ "$HTTP_CODE" -lt 300 ]; then
  SUCCESS_COUNT=$(echo "$BODY" | grep -o '"successCount":[0-9]*' | grep -o '[0-9]*' || echo "?")
  info "Import successful — ${SUCCESS_COUNT} objects imported"
else
  error "Import failed with HTTP ${HTTP_CODE}"
  echo "$BODY"
  exit 1
fi

# -------------------------------------------------------------------
# 4. Print results
# -------------------------------------------------------------------
echo ""
info "Dashboard setup complete!"
echo ""
echo "  Open the dashboard:"
echo "    ${KIBANA_URL}/app/dashboards#/view/llm-observability-dashboard"
echo ""
echo "  Imported objects:"
echo "    - Data view: traces-apm*"
echo "    - Visualization: Cost Distribution by Model (donut)"
echo "    - Visualization: Cost per Call by Model (metric tiles)"
echo "    - Visualization: Cost over Time (line)"
echo "    - Visualization: Token Usage by Model (stacked bar)"
echo "    - Visualization: Input vs Output Token Ratio (bar)"
echo "    - Visualization: Total Tokens by Model (bar)"
echo "    - Visualization: Response Latency by Model (bar)"
echo "    - Visualization: Error Rates by Model (stacked bar)"
echo "    - Dashboard: LLM Observability — Live Metrics (8 panels, 3 rows)"
echo ""
