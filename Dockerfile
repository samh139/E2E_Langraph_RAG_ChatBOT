# 1. Start with the production-grade slim Python image we discussed
FROM python:3.12-slim

# 2. Install essential system utilities needed for networking checks
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container sandbox
WORKDIR /workspace

# 4. Install CPU-only PyTorch in its own layer. This app runs in a container
#    without CUDA; pinning the CPU wheel prevents pip from adding NVIDIA/CUDA
#    runtime packages. Keep this separate so changes to app requirements do
#    not force PyTorch to be resolved and downloaded again.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.13.0+cpu

# 5. Copy requirements separately so Docker can reuse the dependency layer
#    whenever requirements.txt has not changed.
COPY requirements.txt .

# 6. Install app dependencies. The pip cache is shared with the PyTorch step.
RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip \
    pip install -r requirements.txt

## If you want to reinstall all packages
# RUN pip install -r requirements.txt

# 7. Copy the rest of our application code into the workspace
COPY . .

# 8. Keep the container alive using a non-blocking placeholder loop
CMD ["tail", "-f", "/dev/null"]
