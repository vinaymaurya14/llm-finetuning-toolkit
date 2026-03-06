FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Copy application
COPY . .

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
