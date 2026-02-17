.PHONY: up down test test-multiagent dashboard logs clean lint test-unit

up:
	docker compose up --build -d
	@echo "Waiting for services..."
	./scripts/start.sh

down:
	docker compose down

clean:
	docker compose down -v

test:
	./scripts/test-api.sh

test-multiagent:
	./scripts/test-multiagent-menu.sh

test-unit:
	pytest app/tests/ -v

dashboard:
	./kibana/setup-dashboards.sh

logs:
	docker compose logs -f flask-app

lint:
	ruff check app/
