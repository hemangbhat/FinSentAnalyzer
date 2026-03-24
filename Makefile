.PHONY: install run test lint format clean kill help

# ── Help ────────────────────────────────────────────────────────────────────────
help:  ## Show this help message
	@echo off & for /f "tokens=1,2 delims=#" %%a in ('findstr /r "^[a-zA-Z_-]*:.*##" Makefile') do echo %%a %%b

# ── Setup ───────────────────────────────────────────────────────────────────────
install:  ## Create venv and install dependencies
	python -m venv venvMlProject
	venvMlProject\Scripts\pip install --upgrade pip
	venvMlProject\Scripts\pip install -r requirements.txt
	venvMlProject\Scripts\pip install ruff
	@echo.
	@echo ✅ Installation complete.  Activate with:  venvMlProject\Scripts\activate

# ── Run ─────────────────────────────────────────────────────────────────────────
run:  ## Start the Streamlit app
	streamlit run app/app.py

# ── Testing ─────────────────────────────────────────────────────────────────────
test:  ## Run the test suite
	python -m pytest tests/ -v --tb=short

# ── Linting ─────────────────────────────────────────────────────────────────────
lint:  ## Run ruff linter + mypy type checks
	ruff check src/ app/ tests/
	python -m mypy src/utils.py src/preprocess.py --ignore-missing-imports --no-error-summary

format:  ## Auto-format code with ruff
	ruff format src/ app/ tests/
	ruff check --fix src/ app/ tests/

# ── Cleanup ─────────────────────────────────────────────────────────────────────
clean:  ## Remove caches and build artifacts
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist src\__pycache__ rmdir /s /q src\__pycache__
	if exist app\__pycache__ rmdir /s /q app\__pycache__
	if exist tests\__pycache__ rmdir /s /q tests\__pycache__
	@echo ✅ Cleaned.

# ── Port Management ────────────────────────────────────────────────────────────
kill:  ## Kill stale Streamlit processes (Windows)
	powershell -ExecutionPolicy Bypass -File scripts/kill_streamlit.ps1
