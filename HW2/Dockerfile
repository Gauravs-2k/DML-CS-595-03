# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/static

# Expose ports for both web applications
EXPOSE 5000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/web_app/app.py
ENV FLASK_ENV=production

# Create scripts for different deployment options
RUN echo '#!/bin/bash\n\
echo "Starting Federated Learning Simulation..."\n\
python src/federated_learning/fl_simulation.py\n\
echo "Federated Learning Simulation completed!"\n\
echo "Starting Streamlit Web Application..."\n\
streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501' > /app/run_streamlit.sh && chmod +x /app/run_streamlit.sh

RUN echo '#!/bin/bash\n\
echo "Starting Federated Learning Simulation..."\n\
python src/federated_learning/fl_simulation.py\n\
echo "Federated Learning Simulation completed!"\n\
echo "Starting Flask Web Application..."\n\
python src/web_app/app.py' > /app/run_flask.sh && chmod +x /app/run_flask.sh

# Default command - runs FL simulation first, then starts Streamlit app
CMD ["/app/run_streamlit.sh"]

# Alternative commands you can use:
# To run with Streamlit: docker run -p 8501:8501 <image>
# To run with Flask: docker run -p 5000:5000 <image> /app/run_flask.sh
# To run only FL simulation: docker run <image> python src/federated_learning/fl_simulation.py
# To run only Streamlit app: docker run -p 8501:8501 <image> streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
