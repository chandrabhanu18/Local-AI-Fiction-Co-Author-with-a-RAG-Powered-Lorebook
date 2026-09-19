FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
# Pre-install cpu-only torch to keep the image lightweight and reliable
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY docs/ ./docs/
COPY tests/ ./tests/
COPY .env.example .env.example

# Expose API port
EXPOSE 8080

# Run uvicorn server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
