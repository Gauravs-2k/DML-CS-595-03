# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your trained model and app files
COPY models/ ./models/
COPY src/web_app/streamlit_app.py ./
COPY static/ ./static/
COPY data/ ./data/
COPY test/ ./test/
# Expose only Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app

# Single responsibility: Only run the inference app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
