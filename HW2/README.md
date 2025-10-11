# Federated Learning Image Classification Project

This paper presents a comprehensive implementation of federated learning for image classification using the CIFAR-10 dataset with non-IID data distribution across simulated devices. The process incorporates multi-threaded training using ThreadPoolExecutor, AlexNet neural network adaptation, and a Streamlit-based web application for model inference. The federated learning simulation employs 16 clients with Dirichlet-based non-IID data partitioning, achieving approximately 73\% test accuracy while maintaining data privacy through distributed training. The web application enables remote model deployment accessible from any internet-connected device, with successful verification on Chameleon cloud infrastructure. 

## Prerequisites

- **Python Version**: 3.12.2 (verified with `python --version`)
- **Operating System**: Linux/macOS/Windows
- **RAM**: Minimum 8GB recommended
- **GPU**: Optional, but needed for training

## Environment Setup

### 1. Get the Code

**Option A: Extract the Submitted Zip File**
```bash
unzip HW2.zip
cd HW2
```

**Option B: Clone from GitHub Repository**
```bash
git clone https://github.com/Gauravs-2k/DML-CS-595-03.git
cd DML-CS-595-03/HW2
```

### 2. Create Virtual Environment

```bash
python -m venv venv

source venv/bin/activate
```
### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

The CIFAR-10 dataset will be automatically downloaded during the first run of the federated learning simulation.

## Running Federated Learning Simulation

### Local Execution

```bash
source venv/bin/activate

python src/federated_learning/fl_simulation.py
```

**Process:**
- Trains an AlexNet model using federated learning
- Uses 16 clients with non-IID data distribution
- Runs for 15 rounds with 3 epochs per round
- Saves model to `models/federated_model_alexnet_CIFAR10.pth`
- Exports training results to `result/` directory

**Expected output:**
- Training progress with accuracy and loss metrics
- Final model accuracy around 70-75%
- Training visualizations saved to `static/` directory

## Running Web Application

### Local Development

```bash
source venv/bin/activate

streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Access the application at: http://localhost:8501

### Deployment on Chameleon Node

#### 1. Launch on Chameleon Node

```bash

# SSH into Chameleon node
ssh -i your-key.pem ubuntu@<chameleon-node-ip>

# Clone and setup the project
git clone https://github.com/Gauravs-2k/DML-CS-595-03.git
cd DML-CS-595-03/HW2
sudo apt install python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run federated learning simulation
python src/federated_learning/fl_simulation.py
```

#### 2. Launch Web Application

```bash
streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

#### 3. Access from Any Device

- Get the public IP of your Chameleon node
- Access the application at: `http://<chameleon-node-public-ip>:8501`
- The application will be accessible from any Internet-connected device

**Note**: Keep the terminal session running. Use `screen` or `tmux` for persistent sessions:

```bash
sudo apt-get install screen

screen -S fl-app

streamlit run src/web_app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

```

## 📊 Results and Outputs

After running the federated learning simulation:

- **Model**: `models/federated_model_alexnet_CIFAR10.pth`
- **Training Data**: `result/training_results.csv`, `result/training_results.json`
- **Summary**: `result/training_summary.csv`
- **Visualizations**: `static/` directory (data distribution, training history, client performance)

**Typical Performance**:
- Final Test Accuracy: ~73%
- Training Time: ~10-15 minutes (depending on hardware)
- Model Size: ~240MB

## 🐳 Docker Deployment (Alternative)

**Option A: Pull Pre-built Image**
```bash
# Pull the pre-built Docker image
docker pull gauravs2k/federated-learning:latest

# Run the container
docker run -p 8501:8501 gauravs2k/federated-learning:latest
```
**Option B: Build Locally**
```bash
docker build -t federated-learning .

docker run -p 8501:8501 federated-learning
```
Access the application at: http://localhost:8501
