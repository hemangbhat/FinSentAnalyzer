FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_VERBOSITY=error

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than the default CUDA wheel).
# Listing torch in requirements afterwards is then a no-op (already satisfied).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining Python dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

# Copy the nested stock-prediction extension (code + data only, no virtualenvs).
# Required by app/pages/9_Stock_Prediction_Extension.py and src/stock_extension.py.
COPY external-datasets/financial-news-stock-prediction/src/ ./external-datasets/financial-news-stock-prediction/src/
COPY external-datasets/financial-news-stock-prediction/data/ ./external-datasets/financial-news-stock-prediction/data/
COPY external-datasets/financial-news-stock-prediction/models/ ./external-datasets/financial-news-stock-prediction/models/

# Expose Streamlit port
EXPOSE 8501

# Health check (uses Python stdlib — curl is not available in slim images)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
