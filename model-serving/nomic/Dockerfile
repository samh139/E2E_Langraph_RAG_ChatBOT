FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt
COPY serve.py .
EXPOSE 8080
CMD ["gunicorn", "--bind=0.0.0.0:8080", "--workers=1", "--threads=8", "--timeout=120", "serve:app"]
