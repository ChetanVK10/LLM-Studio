# LLMOps Studio Automation Recipes Makefile

.PHONY: install dev test lint format docker-up docker-down clean

# Setup local python environment and install requirements
install:
	python -m venv .venv
	.venv/Scripts/pip install --upgrade pip
	.venv/Scripts/pip install -r requirements.txt

# Start local Streamlit dashboard server
dev:
	.venv/Scripts/streamlit run frontend/app.py

# Run unit tests suite
test:
	.venv/Scripts/pytest tests/

# Execute linter audits
lint:
	.venv/Scripts/ruff check backend/ frontend/

# Format code layout using Black
format:
	.venv/Scripts/black backend/ frontend/

# Build and launch Docker Compose services in background
docker-up:
	docker compose up --build -d

# Stop and teardown Docker Compose services
docker-down:
	docker compose down

# Clear local python cache files
clean:
	rm -rf .pytest_cache/ .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
