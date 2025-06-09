FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install gunicorn
RUN pip install gunicorn

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Copy .env if present (for local dev, not for production best practice)
# COPY .env .env

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "src.api:app"] 