# 1. Start with the production-grade slim Python image we discussed
FROM python:3.12-slim

# 2. Install essential system utilities needed for networking checks
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container sandbox
WORKDIR /workspace

# 4. Copy the requirements file first to take advantage of Docker layer caching
COPY requirements.txt .

# 5. Install all our AI and database client dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of our application code into the workspace
COPY . .

# 7. Keep the container alive using a non-blocking placeholder loop
# This allows us to execute and test scripts manually inside it later
CMD ["tail", "-f", "/dev/null"]