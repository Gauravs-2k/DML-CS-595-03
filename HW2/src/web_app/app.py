"""
Web Application for Federated Learning Model Inference

This Flask web application provides a simple interface for users to upload
images and get predictions from the trained federated learning model.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64

from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model
model = None
device = None
transform = None
class_names = None


def load_model():
    """Load the trained federated learning model"""
    global model, device, transform, class_names
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Look for saved model
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')] if os.path.exists(model_dir) else []
    
    if not model_files:
        raise FileNotFoundError("No trained model found. Please run the federated learning simulation first.")
    
    # Load the most recent model
    model_path = os.path.join(model_dir, model_files[0])
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model info
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    dataset_name = checkpoint['dataset_name']
    
    print(f"Loading {model_name} model trained on {dataset_name} with {num_classes} classes")
    
    # Recreate model architecture
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
        # Adapt for CIFAR input size
        model.features[0] = nn.Conv2d(
            in_channels=3, out_channels=64, 
            kernel_size=3, stride=1, padding=1
        )
        model.classifier[-1] = nn.Linear(
            in_features=4096, out_features=num_classes
        )
    elif model_name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Setup transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Setup class names
    if dataset_name == "CIFAR10":
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    elif dataset_name == "CIFAR100":
        # CIFAR-100 has 100 classes - simplified names
        class_names = [f'class_{i}' for i in range(100)]
    
    print(f"Model loaded successfully on {device}")


def preprocess_image(image):
    """Preprocess uploaded image for model inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device)


def get_prediction(image_tensor):
    """Get model prediction for processed image"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
        
        # Get top 3 predictions
        top_probs, top_classes = torch.topk(probabilities, 3)
        
        predictions = []
        for i in range(3):
            predictions.append({
                'class': class_names[top_classes[i].item()],
                'confidence': float(top_probs[i].item())
            })
    
    return predictions


@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Process image
        image = Image.open(io.BytesIO(file.read()))
        image_tensor = preprocess_image(image)
        
        # Get prediction
        predictions = get_prediction(image_tensor)
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'No model loaded'}), 404
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'device': str(device),
        'num_classes': len(class_names),
        'class_names': class_names,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    })


def create_templates():
    """Create HTML templates directory and files"""
    templates_dir = 'src/web_app/templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html template
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Federated Learning Image Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: border-color 0.3s;
        }
        .upload-area:hover {
            border-color: #007bff;
        }
        .upload-area.dragover {
            border-color: #007bff;
            background-color: #f8f9fa;
        }
        #fileInput {
            display: none;
        }
        .upload-btn {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .upload-btn:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            display: none;
        }
        .result.success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
        }
        .result.error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
        }
        .prediction {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .confidence-bar {
            background-color: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin-top: 5px;
        }
        .confidence-fill {
            background-color: #007bff;
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s;
        }
        .uploaded-image {
            max-width: 200px;
            max-height: 200px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Federated Learning Image Classifier</h1>
        <p style="text-align: center; color: #666;">
            Upload an image to get predictions from our federated learning model
        </p>
        
        <div class="upload-area" id="uploadArea">
            <p>📁 Drag and drop an image here or</p>
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                Choose File
            </button>
            <input type="file" id="fileInput" accept="image/*" onchange="uploadImage()">
            <p style="font-size: 12px; color: #666; margin-top: 10px;">
                Supported formats: PNG, JPG, JPEG, GIF, BMP
            </p>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing image...</p>
        </div>
        
        <div class="result" id="result">
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const resultContent = document.getElementById('resultContent');

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', uploadImage);

        function uploadImage() {
            const file = fileInput.files[0];
            if (file) {
                handleFile(file);
            }
        }

        function handleFile(file) {
            // Validate file type
            const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
            if (!validTypes.includes(file.type)) {
                showError('Please upload a valid image file.');
                return;
            }

            // Validate file size (16MB max)
            if (file.size > 16 * 1024 * 1024) {
                showError('File size too large. Please upload an image smaller than 16MB.');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            // Show loading
            loading.style.display = 'block';
            result.style.display = 'none';

            // Send to server
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                if (data.success) {
                    showPredictions(data);
                } else {
                    showError(data.error || 'Prediction failed');
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                showError('Network error: ' + error.message);
            });
        }

        function showPredictions(data) {
            let html = '<h3>Predictions:</h3>';
            
            // Show uploaded image
            html += `<img src="data:image/png;base64,${data.image}" class="uploaded-image" alt="Uploaded image">`;
            
            // Show predictions
            data.predictions.forEach((pred, index) => {
                const confidence = (pred.confidence * 100).toFixed(1);
                html += `
                    <div class="prediction">
                        <strong>${index + 1}. ${pred.class}</strong>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                        <small>${confidence}% confidence</small>
                    </div>
                `;
            });

            resultContent.innerHTML = html;
            result.className = 'result success';
            result.style.display = 'block';
        }

        function showError(message) {
            resultContent.innerHTML = `<strong>Error:</strong> ${message}`;
            result.className = 'result error';
            result.style.display = 'block';
        }
    </script>
</body>
</html>'''
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)


if __name__ == '__main__':
    try:
        # Create templates
        create_templates()
        
        # Load model
        load_model()
        
        # Run Flask app
        print("Starting web application...")
        print("Open http://localhost:5000 in your browser")
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the federated learning simulation first to train a model.")
    except Exception as e:
        print(f"Error starting web application: {e}")