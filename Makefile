.PHONY: install lint format format-check typecheck test check db-up db-down

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

check: lint format-check typecheck test

# Local PostgreSQL with pgvector (docker-compose.yml); run before `make check` for db tests.
db-up:
	docker compose up -d --wait

db-down:
	docker compose down
