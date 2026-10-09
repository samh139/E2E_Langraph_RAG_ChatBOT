# 1. Start with the production-grade slim Python image we discussed
FROM python:3.12-slim

# 2. Install essential system utilities needed for networking checks
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container sandbox
WORKDIR /workspace

# The cross-encoder reranker uses PyTorch; pin its CPU wheel so pip does not
# resolve CUDA/NVIDIA runtime packages into this CPU-only application image.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.13.0+cpu

# Copy requirements separately so Docker can reuse the dependency layer
#    whenever requirements.txt has not changed.
COPY requirements.txt .

# Install app dependencies (Sentence Transformers is used only for CrossEncoder reranking).
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install -r requirements.txt

## If you want to reinstall all packages
# RUN pip install -r requirements.txt

# 6. Copy the rest of our application code into the workspace
COPY . .

# 7. Keep the container alive using a non-blocking placeholder loop
CMD ["tail", "-f", "/dev/null"]
