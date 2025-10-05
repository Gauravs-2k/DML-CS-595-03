# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-only PyTorch
COPY requirements.txt .

# Install PyTorch CPU-only (MAJOR SIZE REDUCTION)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy essential files including sample images and training data
COPY src/web_app/streamlit_app.py ./
COPY models/federated_model_alexnet_CIFAR10.pth ./models/
COPY test/ ./test/
COPY result/ ./result/
COPY static/ ./static/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
