#!/bin/bash

# Quick Start Script for LLM Observability Stack with Elasticsearch

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}LLM Observability Quick Start${NC}"
echo -e "${BLUE}with Elasticsearch & Kibana${NC}"
echo -e "${BLUE}================================${NC}\n"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is installed${NC}"

# Check .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env

    echo -e "${RED}⚠ IMPORTANT: Add your API keys to .env${NC}"
    echo -e "Press Enter when ready..."
    read
fi

# Linux-specific: Check vm.max_map_count (required for Elasticsearch)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    current_value=$(sysctl -n vm.max_map_count)
    if [ "$current_value" -lt 262144 ]; then
        echo -e "${YELLOW}Setting vm.max_map_count (requires sudo)${NC}"
        sudo sysctl -w vm.max_map_count=262144
        echo -e "${YELLOW}Making permanent in /etc/sysctl.conf${NC}"
        echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf
    fi
fi

# Make scripts executable
chmod +x test-api.sh stress-test.sh 2>/dev/null || true

# Start services
echo -e "\n${BLUE}Starting services...${NC}\n"

if docker compose version &> /dev/null; then
    docker compose up --build -d
else
    docker-compose up --build -d
fi

echo -e "\n${GREEN}✓ Services started!${NC}\n"
echo -e "${YELLOW}Waiting for services to initialize...${NC}\n"

# Wait for services with health checks
echo -n "Waiting for Elasticsearch... "
for i in {1..60}; do
    if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ready${NC}"
        break
    fi
    sleep 2
    echo -n "."
done

echo -n "Waiting for Kibana... "
for i in {1..90}; do
    if curl -s http://localhost:5601/api/status > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ready${NC}"
        break
    fi
    sleep 2
    echo -n "."
done

echo -n "Waiting for OTel Collector... "
for i in {1..30}; do
    if curl -s http://localhost:13133 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ready${NC}"
        break
    fi
    sleep 2
    echo -n "."
done

echo -n "Waiting for Flask API... "
for i in {1..60}; do
    if curl -s http://localhost:5001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ready${NC}"
        break
    fi
    sleep 2
    echo -n "."
done

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}All services are ready!${NC}"
echo -e "${GREEN}================================${NC}\n"

echo -e "${BLUE}Service URLs:${NC}"
echo -e "  • Flask API:              ${YELLOW}http://localhost:5001${NC}"
echo -e "  • Kibana:                 ${YELLOW}http://localhost:5601${NC}"
echo -e "  • Elasticsearch:          ${YELLOW}http://localhost:9200${NC}\n"

echo -e "${BLUE}Next Steps:${NC}"
echo -e "1. Run tests: ${YELLOW}./test-api.sh${NC}"
echo -e "2. View traces in Kibana: ${YELLOW}http://localhost:5601${NC}"
echo -e "3. Explore Observability: ${YELLOW}http://localhost:5601/app/observability${NC}\n"
