# Federated Learning Image Classification Project

This project implements a federated learning simulation for image classification with a web interface for model inference. The implementation meets all CS 595-003 assignment requirements including ThreadPoolExecutor-based multi-threading, non-IID data distribution, detailed logging, and remote deployment capabilities.

## 🎯 Project Overview

The project consists of two main components:

1. **Federated Learning Simulation**: A Python implementation using ThreadPoolExecutor to simulate 64+ devices training a global model collaboratively
2. **Web Application**: A Streamlit-based interface for uploading images and getting predictions from the trained model

### Option 1: Using Docker (Recommended for Chameleon Cloud)

1. **Build the Docker image:**
   ```bash
   docker build -t federated-learning-app .
   ```

2. **Run with Streamlit (Default):**
   ```bash
   docker run -p 8501:8501 federated-learning-app
   ```

### Option 2: Running Locally

1. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate
   ```
   *Note: Your prompt should show `(venv)` when activated. Use `deactivate` to exit.*

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run federated learning simulation:**
   ```bash
   # Option A: Run directly (recommended)
   python src/federated_learning/fl_simulation.py
   
   # Option B: Run as module
   python -m src.federated_learning.fl_simulation
   ```

4. **Start Streamlit web application:**
   ```bash
   streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```

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

# Log out and back in for group changes to take effect
exit
ssh cc@<your-floating-ip>

# Verify Docker installation
docker --version
```

#### 3. Deploy Application

```bash
# Clone or upload your project to the Chameleon node
# Option A: If using git
git clone <your-repository-url>
cd HW2

# Option B: If uploading files
scp -r /path/to/HW2 cc@<your-floating-ip>:~
ssh cc@<your-floating-ip>
cd HW2

# Build Docker image
docker build -t federated-learning-app .

# Run application (accessible from anywhere on the internet)
docker run -d -p 8501:8501 --name fl-app federated-learning-app

# Check if container is running
docker ps

# View logs
docker logs fl-app
```

#### 4. Access Application

- **Streamlit**: `http://<your-floating-ip>:8501`
- **Flask**: `http://<your-floating-ip>:5000` (if using Flask version)

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
