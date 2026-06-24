# Production Setup

## 1. Local setup
```bash
git clone <your-repo-url>
cd FinSentAnalyzer
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run the app locally
```bash
streamlit run app/app.py
```

## 3. Run training / evaluation
```bash
python src/train.py
python src/evaluate.py
python src/benchmark_finbert.py
```

## 4. Docker build
```bash
docker build -t finsentanalyzer .
docker run -p 8501:8501 finsentanalyzer
```

## 5. Streamlit Cloud deployment
1. Push the repository to GitHub.
2. Make sure `requirements.txt` is correct and all assets are committed.
3. Set the Streamlit entry point to `app/app.py`.
4. Confirm the app starts without manual steps.
5. Add the public URL to the README.

## 6. Render / Railway / Fly.io deployment
- use the Dockerfile
- expose port 8501
- verify environment variables
- mount no secrets directly into code
- test the health endpoint after deploy

## 7. Production checklist
- [ ] all imports resolve
- [ ] model files exist
- [ ] optional dependencies degrade gracefully
- [ ] health check passes
- [ ] page load time is acceptable
- [ ] batch processing does not exceed memory limits
- [ ] no secrets committed
- [ ] README instructions match reality
