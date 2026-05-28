# Use Python 3.12 to match your training environment
FROM python:3.12-slim

WORKDIR /app

# Force IPv4 to prevent network hangs during package downloads
RUN apt-get update -o Acquire::ForceIPv4=true && \
    apt-get install -y -o Acquire::ForceIPv4=true build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]