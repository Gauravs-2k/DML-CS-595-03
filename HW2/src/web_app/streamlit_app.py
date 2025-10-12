import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import plotly.express as px
import os


@st.cache_resource
def load_model():
    """Load the trained federated learning model"""
    # determining device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # checking for model directory
    model_dir = 'models'
    if not os.path.exists(model_dir):
        return None, None, None, None, "No models directory found. Please run the federated learning simulation first."
    
    # listing model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        return None, None, None, None, "No trained model found. Please run the federated learning simulation first."
    
    # selecting most recent model
    model_path = os.path.join(model_dir, model_files[0])
    # loading checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # extracting model info
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    dataset_name = checkpoint['dataset_name']
    
    # recreating model architecture
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
    
    # loading trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    # moving model to device
    model = model.to(device)
    # setting model to eval mode
    model.eval()
    
    # setting up transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # setting up class names
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
    # attempting to load training data
    try:
        # checking if file exists
        if os.path.exists('result/training_results.csv'):
            df = pd.read_csv('result/training_results.csv')
            return df
        return None
    except Exception as e:
        st.error(f"Error loading training data: {e}")
        return None


def preprocess_image(image, transform):
    """Preprocess uploaded image for model inference"""
    # converting image to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # applying transform
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def get_prediction(model, image_tensor, class_names, device):
    """Get model prediction for processed image"""
    # moving tensor to device
    image_tensor = image_tensor.to(device)
    
    # performing inference
    with torch.no_grad():
        # getting model outputs
        outputs = model(image_tensor)
        # computing probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # getting top predictions
        top_probs, top_classes = torch.topk(probabilities, min(5, len(class_names)))
        
        # initializing predictions list
        predictions = []
        # building predictions list
        for i in range(len(top_probs)):
            predictions.append({
                'class': class_names[top_classes[i].item()],
                'confidence': float(top_probs[i].item()),
                'percentage': float(top_probs[i].item()) * 100
            })
    
    return predictions


def get_sample_images():
    # setting sample directory
    sample_dir = 'test'
    # checking if directory exists
    if not os.path.exists(sample_dir):
        return []
    # defining valid extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    # initializing files list
    files = []
    # collecting valid files
    for filename in sorted(os.listdir(sample_dir)):
        if filename.lower().endswith(valid_extensions):
            files.append(os.path.join(sample_dir, filename))
    return files


def set_current_image(image, name):
    # converting to RGB
    img = image.convert('RGB')
    # clearing previous image
    if 'current_image' in st.session_state:
        del st.session_state['current_image']
    # setting current image
    st.session_state.current_image = img.copy()
    # setting image name
    st.session_state.current_image_name = name
    # clearing predictions
    st.session_state.predictions = None


def main():
    # setting page config
    st.set_page_config(
        page_title="Federated Learning Image Classifier",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Federated Learning Image Classifier")
    st.markdown("Upload an image to get predictions from our federated learning model")
    
    # loading model
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
        # selecting image source
        source = st.radio("Image source", ["Upload", "Sample"], horizontal=True)
        current_image = None
        current_name = None
        # handling upload
        if source == "Upload":
            # getting uploaded file
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image to classify"
            )
            # processing uploaded file
            if uploaded_file is not None:
                # opening image
                image = Image.open(uploaded_file)
                # setting current image
                set_current_image(image, uploaded_file.name)
        else:
            # handling sample images
            if sample_images:
                # getting sample names
                sample_names = [os.path.basename(path) for path in sample_images]
                # initializing selection
                if 'selected_sample' not in st.session_state:
                    st.session_state.selected_sample = sample_names[0]
                    st.session_state.loaded_sample = None

                # selecting sample
                selected_sample = st.selectbox(
                    "Sample images",
                    sample_names,
                    index=sample_names.index(st.session_state.selected_sample)
                )
                st.session_state.selected_sample = selected_sample

                # checking if load needed
                needs_load = st.session_state.loaded_sample != selected_sample
                # loading sample image
                if needs_load:
                    # getting sample path
                    sample_path = sample_images[sample_names.index(selected_sample)]
                    # opening sample image
                    image = Image.open(sample_path)
                    # setting current image
                    set_current_image(image, selected_sample)
                    st.session_state.loaded_sample = selected_sample
            else:
                st.info("No sample images available")

        # getting current image
        current_image = st.session_state.current_image
        current_name = st.session_state.current_image_name

        # displaying image
        if current_image is not None:
            st.image(current_image, caption=current_name, use_column_width=True)
        # checking if classification disabled
        classify_disabled = current_image is None
        # handling classify button
        if st.button("Classify Image", type="primary", disabled=classify_disabled):
            # showing spinner
            with st.spinner("Processing image..."):
                # attempting prediction
                try:
                    # preprocessing image
                    image_tensor = preprocess_image(current_image, transform)
                    # getting predictions
                    predictions = get_prediction(model, image_tensor, class_names, device)
                    # storing predictions
                    st.session_state.predictions = predictions
                # handling error
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    
    with col2:
        st.header("Predictions")
        
        # checking for predictions
        if hasattr(st.session_state, 'predictions') and st.session_state.predictions:
            # getting predictions
            predictions = st.session_state.predictions

            # getting top prediction
            top_pred = predictions[0]
            # displaying top prediction
            st.success(f"**Top Prediction: {top_pred['class']}** ({top_pred['percentage']:.1f}% confidence)")

            # creating dataframe
            pred_df = pd.DataFrame(predictions)

            # creating bar chart
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
            # displaying chart
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detailed Results")
            # displaying each prediction
            for i, pred in enumerate(predictions):
                # expanding details
                with st.expander(f"{i+1}. {pred['class']} - {pred['percentage']:.1f}%"):
                    st.write(f"**Confidence:** {pred['confidence']:.4f}")
                    st.write(f"**Percentage:** {pred['percentage']:.2f}%")
        else:
            st.info("Upload an image and click 'Classify Image' to see predictions.")
    
    # training Results Visualization
    st.header("📊 Training Results")
    
    # loading training data
    training_data = load_training_data()
    # if data exists
    if training_data is not None:
        # creating tabs
        tab1, tab2, tab3 = st.tabs(["Training Overview", "Per-Client Analysis", "Raw Data"])
        
        with tab1:
            # creating columns
            col1, col2 = st.columns(2)
            
            with col1:
                # aggregating round data
                round_summary = training_data.groupby('round').agg({
                    'train_loss': 'mean',
                    'train_acc': 'mean'
                }).reset_index()
                
                # creating loss plot
                fig = px.line(round_summary, x='round', y='train_loss', 
                            title='Average Training Loss per Round')
                # displaying loss plot
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # creating accuracy plot
                fig = px.line(round_summary, x='round', y='train_acc', 
                            title='Average Training Accuracy per Round')
                # displaying accuracy plot
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Client Performance Analysis")
            
            # aggregating client data
            client_summary = training_data.groupby('client_id').agg({
                'train_loss': ['mean', 'std'],
                'train_acc': ['mean', 'std']
            }).round(4)
            
            # displaying summary
            st.dataframe(client_summary, use_container_width=True)
            
            # creating box plot
            fig = px.box(training_data, x='client_id', y='train_loss', 
                        title='Training Loss Distribution by Client')
            fig.update_xaxes(type='category')
            # displaying box plot
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Raw Training Data")
            # showing record count
            st.write(f"Total records: {len(training_data)}")
            # displaying data
            st.dataframe(training_data.head(1000), use_container_width=True)  # Show first 1000 rows
            
            # preparing csv
            csv = training_data.to_csv(index=False)
            # download button
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