"""
Streamlit Web Application for Federated Learning Model Inference

This Streamlit app provides an intuitive interface for users to upload
images and get predictions from the trained federated learning model.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import io


@st.cache_resource
def load_model():
    """Load the trained federated learning model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Look for saved model
    model_dir = 'models'
    if not os.path.exists(model_dir):
        return None, None, None, None, "No models directory found. Please run the federated learning simulation first."
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        return None, None, None, None, "No trained model found. Please run the federated learning simulation first."
    
    # Load the most recent model
    model_path = os.path.join(model_dir, model_files[0])
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model info
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    dataset_name = checkpoint['dataset_name']
    
    # Recreate model architecture
    if model_name == "alexnet":
        model = models.alexnet(weights=None)
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
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
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
        class_names = [f'class_{i}' for i in range(100)]
    
    return model, transform, class_names, device, None


@st.cache_data
def load_training_data():
    """Load training results for visualization"""
    try:
        if os.path.exists('result/training_results.csv'):
            df = pd.read_csv('result/training_results.csv')
            return df
        return None
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None


def preprocess_image(image, transform):
    """Preprocess uploaded image for model inference"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def get_prediction(model, image_tensor, class_names, device):
    """Get model prediction for processed image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 5 predictions
        top_probs, top_classes = torch.topk(probabilities, min(5, len(class_names)))
        
        predictions = []
        for i in range(len(top_probs)):
            predictions.append({
                'class': class_names[top_classes[i].item()],
                'confidence': float(top_probs[i].item()),
                'percentage': float(top_probs[i].item()) * 100
            })
    
    return predictions


def get_sample_images():
    sample_dir = 'test'
    if not os.path.exists(sample_dir):
        return []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    files = []
    for filename in sorted(os.listdir(sample_dir)):
        if filename.lower().endswith(valid_extensions):
            files.append(os.path.join(sample_dir, filename))
    return files


def set_current_image(image, name):
    img = image.convert('RGB')
    if 'current_image' in st.session_state:
        del st.session_state['current_image']
    st.session_state.current_image = img.copy()
    st.session_state.current_image_name = name
    st.session_state.predictions = None


def main():
    st.set_page_config(
        page_title="Federated Learning Image Classifier",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Federated Learning Image Classifier")
    st.markdown("Upload an image to get predictions from our federated learning model")
    
    # Load model
    model, transform, class_names, device, error = load_model()
    
    if error:
        st.error(error)
        st.info("Please run the federated learning simulation first to train a model.")
        return
    
    # Sidebar with model information
    with st.sidebar:
        st.header("Model Information")
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            st.write(f"**Device:** {device}")
            st.write(f"**Classes:** {len(class_names)}")
            st.write(f"**Total Parameters:** {total_params:,}")
            st.write(f"**Trainable Parameters:** {trainable_params:,}")
            
            st.subheader("Class Names")
            for i, class_name in enumerate(class_names):
                st.write(f"{i}: {class_name}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])

    sample_images = get_sample_images()
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
        st.session_state.current_image_name = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    with col1:
        st.header("Select Image")
        source = st.radio("Image source", ["Upload", "Sample"], horizontal=True)
        current_image = None
        current_name = None
        if source == "Upload":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image to classify"
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                set_current_image(image, uploaded_file.name)
        else:
            if sample_images:
                sample_names = [os.path.basename(path) for path in sample_images]
                if 'selected_sample' not in st.session_state:
                    st.session_state.selected_sample = sample_names[0]
                    st.session_state.loaded_sample = None

                selected_sample = st.selectbox(
                    "Sample images",
                    sample_names,
                    index=sample_names.index(st.session_state.selected_sample)
                )
                st.session_state.selected_sample = selected_sample

                needs_load = st.session_state.loaded_sample != selected_sample
                if needs_load:
                    sample_path = sample_images[sample_names.index(selected_sample)]
                    image = Image.open(sample_path)
                    set_current_image(image, selected_sample)
                    st.session_state.loaded_sample = selected_sample
            else:
                st.info("No sample images available")

        current_image = st.session_state.current_image
        current_name = st.session_state.current_image_name

        if current_image is not None:
            st.image(current_image, caption=current_name, use_column_width=True)
        classify_disabled = current_image is None
        if st.button("Classify Image", type="primary", disabled=classify_disabled):
            with st.spinner("Processing image..."):
                try:
                    image_tensor = preprocess_image(current_image, transform)
                    predictions = get_prediction(model, image_tensor, class_names, device)
                    st.session_state.predictions = predictions
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    
    with col2:
        st.header("Predictions")
        
        if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
            predictions = st.session_state.predictions

            top_pred = predictions[0]
            st.success(f"**Top Prediction: {top_pred['class']}** ({top_pred['percentage']:.1f}% confidence)")

            pred_df = pd.DataFrame(predictions)

            fig = px.bar(
                pred_df,
                x='percentage',
                y='class',
                orientation='h',
                title="Confidence Scores",
                labels={'percentage': 'Confidence (%)', 'class': 'Class'},
                color='percentage',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detailed Results")
            for i, pred in enumerate(predictions):
                with st.expander(f"{i+1}. {pred['class']} - {pred['percentage']:.1f}%"):
                    st.write(f"**Confidence:** {pred['confidence']:.4f}")
                    st.write(f"**Percentage:** {pred['percentage']:.2f}%")
        else:
            st.info("Upload an image and click 'Classify Image' to see predictions.")
    
    # Training Results Visualization
    st.header("📊 Training Results")
    
    training_data = load_training_data()
    if training_data is not None:
        tab1, tab2, tab3 = st.tabs(["Training Overview", "Per-Client Analysis", "Raw Data"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Training loss over rounds
                round_summary = training_data.groupby('round').agg({
                    'train_loss': 'mean',
                    'train_acc': 'mean'
                }).reset_index()
                
                fig = px.line(round_summary, x='round', y='train_loss', 
                            title='Average Training Loss per Round')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Training accuracy over rounds
                fig = px.line(round_summary, x='round', y='train_acc', 
                            title='Average Training Accuracy per Round')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Per-client analysis
            st.subheader("Client Performance Analysis")
            
            client_summary = training_data.groupby('client_id').agg({
                'train_loss': ['mean', 'std'],
                'train_acc': ['mean', 'std']
            }).round(4)
            
            st.dataframe(client_summary, use_container_width=True)
            
            # Client loss distribution
            fig = px.box(training_data, x='client_id', y='train_loss', 
                        title='Training Loss Distribution by Client')
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Raw Training Data")
            st.write(f"Total records: {len(training_data)}")
            st.dataframe(training_data.head(1000), use_container_width=True)  # Show first 1000 rows
            
            # Download button for full data
            csv = training_data.to_csv(index=False)
            st.download_button(
                label="Download Full Training Data",
                data=csv,
                file_name="training_results.csv",
                mime="text/csv"
            )
    else:
        st.info("No training data found. Run the federated learning simulation to generate training results.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Federated Learning Image Classifier** - CS 595-003 Decentralized ML Systems")


if __name__ == "__main__":
    main()