FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir flask python-dotenv requests google-genai openai anthropic

COPY solver.py app.py Solutions.md ./
COPY templates/ templates/
COPY data/ data/

# Cloud Run uses PORT env var (default 8080)
ENV PORT=8080

EXPOSE 8080

CMD ["python", "-c", "from app import app; app.run(host='0.0.0.0', port=int(__import__('os').environ.get('PORT', 8080)), debug=False, threaded=True)"]
