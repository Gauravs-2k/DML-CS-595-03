# Federated Learning Image Classification Project

This project implements a federated learning simulation for image classification with a web interface for model inference. The implementation meets all CS 595-003 assignment requirements including ThreadPoolExecutor-based multi-threading, non-IID data distribution, detailed logging, and remote deployment capabilities.

## 🎯 Project Overview

The project consists of two main components:

✅ Federated Learning Simulation: COMPLETED - A Python implementation using ThreadPoolExecutor to simulate 16 devices training a global model collaboratively (achieved 73.40% accuracy)

🔄 Web Application: IN PROGRESS - A Streamlit-based interface for uploading images and getting predictions from the trained model

## 📋 Current Status

✅ COMPLETED Components
- Federated Learning Training - 16 clients, 15 rounds, 73.40% final accuracy
- Model Saved - models/federated_model_alexnet_CIFAR10.pth
- Training Data Export - CSV files with all required columns
- Streamlit App - Working locally for inference
- Data Visualizations - Training curves and data distribution plots

🔄 TODO Components (Due Tomorrow 11:59 PM)
- Docker Containerization - Package Streamlit app for deployment
- Chameleon Cloud Deployment - Deploy container with floating IP access
- IEEE Report - 2-page academic paper with results
- Final Submission - Package all deliverables

## 🚀 Quick Start

### Option 1: Local Testing (WORKING)
Run Streamlit interface:

```bash
# Activate virtual environment
source venv/bin/activate

# Start Streamlit app
streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Access at: http://localhost:8501

### Option 2: Docker Deployment (NEEDS COMPLETION)
Build Docker image:

```bash
docker build -t federated-inference .
```

Run container:

```bash
docker run -p 8501:8501 federated-inference
```

## 🐳 Updated Dockerfile (Inference Only)

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/web_app/streamlit_app.py ./
COPY models/ ./models/
COPY static/ ./static/
COPY data/ ./data/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit app (inference only)
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

## 🌐 Chameleon Cloud Deployment Steps

### 1. Prerequisites
- Reserve Chameleon Cloud lease with Floating IP
- Use CC-Ubuntu20.04 image
- Configure security groups for port 8501

### 2. Install Docker on Chameleon

```bash
# SSH into Chameleon instance
ssh cc@<your-floating-ip>

# Install Docker
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Re-login to apply docker group
exit && ssh cc@<your-floating-ip>
```

### 3. Deploy Application

```bash
# Upload your project files
scp -r HW2/ cc@<your-floating-ip>:~

# SSH and deploy
ssh cc@<your-floating-ip>
cd HW2

# Build and run
docker build -t federated-inference .
docker run -d -p 8501:8501 --name fl-app federated-inference

# Verify deployment
docker ps
docker logs fl-app
```

### 4. Test Remote Access
Access at: http://<your-floating-ip>:8501

Take screenshot for IEEE report

## 🤖 Federated Learning Results (COMPLETED)

### Training Configuration

```python
config = {
    'num_clients': 16,           # ✅ Meets requirement (≥16)
    'dataset_name': 'CIFAR10',   # ✅ Standard dataset
    'model_name': 'alexnet',     # ✅ ImageNet architecture adapted
    'max_workers': 8,            # ✅ ThreadPoolExecutor
    'epochs_per_round': 3,       # ✅ Local training epochs
    'num_rounds': 15,            # ✅ Communication rounds
    'learning_rate': 0.01        # ✅ Optimized learning rate
}
```

### Training Results
- Initial Accuracy: 10.00% (random baseline)
- Final Accuracy: 73.40% (excellent performance)
- Training Time: ~45 minutes on GPU
- Data Records: 70,740 training logs exported

### Required Data Export ✅

The `data/training_results.csv` contains all required columns:

```
time,round,batch_num,client_id,train_loss,train_acc
2025-10-05 14:00:00,1,0,0,1.6236,0.41
2025-10-05 14:00:01,1,1,1,1.5892,0.43
...
```

## 🌐 Streamlit Interface Features

### Working Features ✅
- Image Upload: Drag-and-drop interface for PNG, JPG, JPEG, GIF, BMP
- Model Inference: Real-time predictions on uploaded images
- CIFAR-10 Classes: Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Training Analytics: Interactive charts showing federated learning progress
- Model Information: Architecture details and performance metrics

### Interface Sections
- Model Information - Device, classes, parameters
- Image Classification - Upload and classify images
- Training Results - Visualization of FL performance
- Raw Data - Access to training logs

## 📊 Assignment Deliverables Status

### ✅ COMPLETED
- FL Simulation Code - Working with ThreadPoolExecutor
- Non-IID Data Distribution - Visualization saved
- Model Training - 73.40% accuracy achieved
- Training Data CSV - All required columns present
- Streamlit Web App - Working locally

### 🔄 IN PROGRESS (Final Steps)
- Docker Container - Needs testing and deployment
- Chameleon Deployment - Remote access setup
- IEEE Report - 2-page paper with three required figures
- Final Packaging - ZIP file with all components

## 🕒 Remaining Timeline (Due Oct 6, 11:59 PM)

### Priority 1 (Next 2 hours): Docker & Deployment
- Fix Dockerfile for inference-only container
- Test Docker build and run locally
- Deploy on Chameleon with floating IP
- Verify remote access and take screenshots

### Priority 2 (Next 3 hours): IEEE Report
- Write 2-page report with required sections
- Include three required figures:
  - Learning curve (training accuracy/loss progression)
  - Data distribution plot (non-IID visualization)
  - Web application screenshot (successful inference)

### Priority 3 (Final hour): Submission
- Package source code, Dockerfile, README
- Include CSV training data
- Submit ZIP file under 10MB

## 🐛 Known Working Components
- Local FL Training: ✅ 73.40% accuracy achieved
- Local Streamlit App: ✅ Inference working
- Data Export: ✅ CSV with all required columns
- Model Saving: ✅ AlexNet model saved properly

## 📚 Technical Stack
- Python 3.11 (verified working)
- PyTorch: Model training and inference
- Streamlit: Web interface
- ThreadPoolExecutor: Concurrent FL simulation
- Docker: Containerization for deployment
- Chameleon Cloud: Remote deployment platform

---

**Course**: CS 595-003 Decentralized ML Systems  
**Assignment**: HW2 - Federated Learning and AI Model Serving  
**Instructor**: Dr. Nathaniel Hudson

**Status**: Training ✅ Complete | Deployment 🔄 In Progress | Due: Oct 6, 11:59 PM

## 🌐 Chameleon Cloud Deployment

### Prerequisites

1. **Chameleon Cloud Account**: Ensure you have access to Chameleon Cloud
2. **Floating IP**: Reserve a floating IP address for external access
3. **Security Groups**: Configure security groups to allow HTTP traffic

### Step-by-Step Deployment

#### 1. Create and Configure Chameleon Instance

```bash
# Launch instance with CC-Ubuntu20.04 image
# Reserve a floating IP and associate it with your instance
# Configure security groups to allow ports 8501 (Streamlit) or 5000 (Flask)
```

#### 2. Install Docker on Chameleon Node

```bash
# SSH into your Chameleon instance
ssh cc@<your-floating-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

exit
ssh cc@<your-floating-ip>

# Verify Docker installation
docker --version
```

#### 3. Deploy Application

```bash
git clone <your-repository-url>
cd HW2

# Option B: If uploading files
scp -r /path/to/HW2 cc@<your-floating-ip>:~
ssh cc@<your-floating-ip>
cd HW2

# Build Docker image
docker build -t federated-inference .

docker run -p 8501:8501 federated-inference


# Check if container is running
docker ps

# View logs
docker logs fl-app
```

#### 4. Access Application

- **Streamlit**: `http://<your-floating-ip>:8501`

### Important Deployment Notes

- **Binding Address**: The application binds to `0.0.0.0` to accept connections from any IP
- **Firewall**: Ensure Chameleon security groups allow inbound traffic on ports 8501/5000
- **Resource Requirements**: The FL simulation may take 10-30 minutes depending on hardware
- **Persistent Storage**: Use Docker volumes if you need to persist data across container restarts

### Docker Commands for Chameleon

```bash
# Full deployment (FL simulation + Streamlit)
docker run -d -p 8501:8501 --name fl-app federated-learning-app

# Only run FL simulation (for training models)
docker run --name fl-train federated-learning-app python src/federated_learning/fl_simulation.py

# Only run web app (if model already exists)
docker run -d -p 8501:8501 --name fl-web federated-learning-app streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

# Copy trained model from container
docker cp fl-train:/app/models ./models

# Run with persistent storage
docker run -d -p 8501:8501 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data --name fl-app federated-learning-app
```

## 🤖 Federated Learning Implementation

### Assignment Requirements Compliance

- ✅ **Minimum 16 Devices**: Configurable (default: 64)
- ✅ **ThreadPoolExecutor**: Used for concurrent local training
- ✅ **ImageNet Architecture**: AlexNet/ResNet18 adapted for CIFAR
- ✅ **Non-IID Data**: Dirichlet distribution partitioning with visualization
- ✅ **Model Saving**: Final global model saved to `models/`
- ✅ **Detailed Logging**: CSV export with columns: time, round, batch_num, client_id, train_loss, train_acc
- ✅ **Web Application**: Streamlit interface with image upload and inference
- ✅ **Docker Support**: Complete containerization for Chameleon deployment

### Detailed Logging Output

The simulation generates comprehensive training logs as required:

```csv
time,round,batch_num,client_id,epoch,train_loss,train_acc,batch_size
2025-09-21 10:30:15,1,0,0,0,2.3456,0.12,32
2025-09-21 10:30:15,1,1,0,0,2.2134,0.18,32
...
```

Files generated:
- `data/training_results.csv` - Detailed per-batch training logs
- `data/training_results.json` - Same data in JSON format
- `data/training_results.parquet` - Same data in Parquet format
- `data/training_summary.csv` - Per-round summary statistics

### Configuration

Edit the configuration in `src/federated_learning/fl_simulation.py`:

```python
config = {
    'num_clients': 64,              # Number of simulated devices (min: 16)
    'dataset_name': 'CIFAR10',      # 'CIFAR10' or 'CIFAR100'
    'model_name': 'alexnet',        # 'alexnet' or 'resnet18'
    'max_workers': 8,               # ThreadPoolExecutor workers
    'epochs_per_round': 3,          # Local training epochs
    'num_rounds': 15,               # Federated learning rounds
    'learning_rate': 0.001
}
```

## 🌐 Web Application Features

### Streamlit Interface (Recommended)

- **Interactive UI**: Modern, responsive design with drag-and-drop upload
- **Real-time Prediction**: Shows top 5 predictions with confidence visualization
- **Training Analytics**: Built-in charts for training loss, accuracy, and client analysis
- **Model Information**: Detailed model statistics and class information
- **Data Export**: Download training results directly from the interface

### Flask Interface (Alternative)

- **REST API**: Simple endpoints for programmatic access
- **JSON Responses**: Machine-readable prediction results
- **Health Checks**: Monitor application status

### Key Features

- **Multiple Format Support**: PNG, JPG, JPEG, GIF, BMP
- **Confidence Visualization**: Interactive charts showing prediction confidence
- **Training History**: Real-time visualization of federated learning progress
- **Client Analysis**: Per-client performance metrics and statistics

## 🔧 Development & Testing

### Local Development

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FL simulation
python src/federated_learning/fl_simulation.py

# Test Streamlit app locally
streamlit run src/web_app/streamlit_app.py

# Test Flask app locally
python src/web_app/app.py
```

### Testing Docker Build

```bash
# Build image
docker build -t federated-learning-app .

# Test locally
docker run -p 8501:8501 federated-learning-app

# Test different components
docker run federated-learning-app python src/federated_learning/fl_simulation.py
```

## 📊 Expected Results

### CIFAR-10 Performance
- **Initial Accuracy**: ~10% (random baseline)
- **Final Accuracy**: 60-80% (after 15 rounds)
- **Training Time**: 10-30 minutes (depending on hardware)
- **Data Export**: ~50,000+ training records (64 clients × 15 rounds × batches)

### Generated Outputs

1. **Model Files**: `models/federated_model_*.pth`
2. **Training Data**: `data/training_results.csv` (required format)
3. **Visualizations**: 
   - `static/data_distribution.png` (non-IID distribution heatmap)
   - `static/training_history.png` (accuracy/loss curves)

## 🐛 Troubleshooting

### Common Issues

1. **"No trained model found"**
   - Solution: Run FL simulation first
   - Check: `models/` directory exists with `.pth` files

2. **Docker container exits**
   - Check logs: `docker logs <container-name>`
   - Verify: All dependencies installed correctly

3. **Cannot access from external IP**
   - Verify: Application binds to `0.0.0.0`
   - Check: Chameleon security groups allow port access
   - Confirm: Floating IP properly associated

4. **Out of Memory during training**
   - Reduce: `max_workers` in configuration
   - Use: Smaller batch sizes or fewer clients

### Chameleon-Specific Issues

1. **Docker permission denied**
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. **Port not accessible**
   ```bash
   # Check if port is open
   sudo netstat -tlnp | grep :8501
   
   # Check security groups in Chameleon dashboard
   ```

3. **Container memory limits**
   ```bash
   # Run with memory limits
   docker run -m 4g -p 8501:8501 federated-learning-app
   ```

## � Assignment Deliverables

This implementation provides all required deliverables:

1. **Source Code**: Complete FL simulation and web application
2. **requirements.txt**: All Python dependencies
3. **Dockerfile**: Container configuration for deployment
4. **README**: This comprehensive documentation
5. **Training Data**: CSV files with required columns
6. **Visualizations**: Data distribution and training curves

### Required Data Columns ✅

The exported `data/training_results.csv` contains:
- `time`: Timestamp of training event
- `round`: Federated learning round number
- `batch_num`: Batch number within client's local training
- `client_id`: Unique identifier for simulated device
- `train_loss`: Training loss for the batch
- `train_acc`: Training accuracy for the batch
- Additional columns: `epoch`, `batch_size` for completeness

## 🚀 Python Version & Environment

**Python 3.11.x** (verified with `python --version`)

### Virtual Environment Setup

It is highly recommended to use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### PyTorch Installation Issues

If you encounter PyTorch/torchvision compatibility issues (common with Python 3.12+), reinstall with compatible versions:

```bash
# Remove existing installations
pip uninstall torch torchvision torchaudio

# Install compatible versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📚 References

- McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data.
- PyTorch Documentation: https://pytorch.org/docs/
- Streamlit Documentation: https://docs.streamlit.io/
- Chameleon Cloud Documentation: https://chameleoncloud.readthedocs.io/

---

**Course**: CS 595-003 Decentralized ML Systems  
**Assignment**: HW2 - Federated Learning and AI Model Serving  
**Instructor**: Dr. Nathaniel Hudson
