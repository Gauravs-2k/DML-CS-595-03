# Federated Learning Image Classification Project

This project implements a federated learning simulation for image classification with a web interface for model inference. The implementation meets all CS 595-003 assignment requirements including ThreadPoolExecutor-based multi-threading, non-IID data distribution, detailed logging, and remote deployment capabilities.

## 🚀 Quick Start

### Run Federated Learning Training

```bash
# Activate virtual environment
source venv/bin/activate

# Run the federated learning simulation
python src/federated_learning/fl_simulation.py
```

### Run Streamlit Web App Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Start Streamlit app
streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Access at: http://localhost:8501

### Run with Docker (Inference Only)

```bash
# Pull the pre-built Docker image
docker pull gauravs2k/federated-learning:latest

# Run the container
docker run -p 8501:8501 gauravs2k/federated-learning:latest
```

Access at: http://localhost:8501

## 📊 Results

- **Final Accuracy**: 73.40% on CIFAR-10
- **Training Data**: Exported to `result/training_results.csv`
- **Model**: Saved as `models/federated_model_alexnet_CIFAR10.pth`

## 🐳 Docker Image

The Docker image contains:
- Pre-trained federated learning model
- Streamlit web interface
- Sample test images
- Training results and visualizations

**Course**: CS 595-003 Decentralized ML Systems  
**Assignment**: HW2 - Federated Learning and AI Model Serving  
**Instructor**: Dr. Nathaniel Hudson
