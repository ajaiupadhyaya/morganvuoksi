# Variables
PYTHON = python3
PIP = pip3
VENV = .venv
PYTEST = pytest
COVERAGE = coverage

# Default target
all: install test lint

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip

# Install dependencies
install: venv
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	. $(VENV)/bin/activate && $(PIP) install -r requirements-dev.txt

# Run tests
test:
	. $(VENV)/bin/activate && $(PYTEST) tests/ -v

# Run tests with coverage
coverage:
	. $(VENV)/bin/activate && $(COVERAGE) run -m pytest tests/
	. $(VENV)/bin/activate && $(COVERAGE) report
	. $(VENV)/bin/activate && $(COVERAGE) html

# Run linter
lint:
	. $(VENV)/bin/activate && flake8 src/ tests/
	. $(VENV)/bin/activate && black src/ tests/
	. $(VENV)/bin/activate && isort src/ tests/

# Run type checker
typecheck:
	. $(VENV)/bin/activate && mypy src/ tests/

# Run security checks
security:
	. $(VENV)/bin/activate && bandit -r src/
	. $(VENV)/bin/activate && safety check

# Clean up
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run the system
run:
	. $(VENV)/bin/activate && $(PYTHON) run_all.py

# Run backtest
backtest:
	. $(VENV)/bin/activate && $(PYTHON) run_backtest.py

# Run dashboard
dashboard:
	. $(VENV)/bin/activate && streamlit run src/dashboard/app.py

# Build Docker image
docker-build:
	docker build -t quant-finance .

# Run Docker container
docker-run:
	docker run -p 8501:8501 quant-finance

# Deploy to production
deploy:
	. $(VENV)/bin/activate && $(PYTHON) -m src.deployment.deploy

# Help
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test      - Run tests"
	@echo "  make coverage  - Run tests with coverage"
	@echo "  make lint      - Run linter"
	@echo "  make typecheck - Run type checker"
	@echo "  make security  - Run security checks"
	@echo "  make clean     - Clean up"
	@echo "  make run       - Run the system"
	@echo "  make backtest  - Run backtest"
	@echo "  make dashboard - Run dashboard"
	@echo "  make deploy    - Deploy to production"

.PHONY: all venv install test coverage lint typecheck security clean run backtest dashboard docker-build docker-run deploy help 