# 1. Start with the production-grade slim Python image
FROM python:3.12-slim

# 2. Install essential system utilities needed for networking checks
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /workspace

# 4. Install CPU-only PyTorch wheel without BuildKit cache mounts
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0+cpu

# 5. Copy requirements separately so Docker can reuse the dependency layer
COPY requirements.txt .

# 6. Install app dependencies
RUN pip --timeout 300 --retries 10 install --no-cache-dir -r requirements.txt

# 7. Download the spaCy English model for Presidio
RUN python -m spacy download en_core_web_sm

# 8. Copy the rest of your application code
COPY . .

# 9. Expose Cloud Run's port and start the FastAPI WebSocket service
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.websocket_service.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
