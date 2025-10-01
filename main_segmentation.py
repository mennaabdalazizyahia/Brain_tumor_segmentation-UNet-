import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Image Segmentation", layout="wide")
st.title("🖼️ Image Segmentation App")
st.write("Upload your model and image for segmentation")

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=[1,2,3])
    union = tf.keras.backend.sum(y_true,[1,2,3]) + tf.keras.backend.sum(y_pred,[1,2,3]) - intersection
    iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def f1_score(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+tf.keras.backend.epsilon())
    return f1_val

st.sidebar.header("📁 Step 1: Upload Model")
uploaded_model = st.sidebar.file_uploader(
    "Upload your trained model (.h5 or .keras)", 
    type=['h5', 'keras'],
    help="Upload the segmentation model file you trained"
)

@st.cache_resource
def load_uploaded_model(model_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_file.write(model_file.getvalue())
            tmp_path = tmp_file.name
        
        model = tf.keras.models.load_model(
            tmp_path, 
            custom_objects={
                'dice_coef': dice_coef,
                'iou_coef': iou_coef,
                'f1_score': f1_score
            }
        )
        
        os.unlink(tmp_path)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[:, :, :3]
    return np.expand_dims(image_array, axis=0)

def create_overlay(original, mask, alpha=0.5):
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
    mask_colored = np.zeros_like(original)
    mask_colored[:, :, 1] = mask_resized * 255
    overlay = cv2.addWeighted(original, 1 - alpha, mask_colored, alpha, 0)
    return overlay

def main():
    model = None
    
    if uploaded_model is not None:
        with st.spinner("Loading model..."):
            model = load_uploaded_model(uploaded_model)
        
        if model is not None:
            st.sidebar.success("✅ Model loaded successfully!")
            
            st.header("Step 2: Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image for segmentation", 
                type=['jpg', 'jpeg', 'png','tif','tiff','TIF','TIFF']
            )
            
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Run Segmentation", type="primary"):
                    with st.spinner("Processing image..."):
                        try:
                            processed_image = preprocess_image(image)
                            original_array = np.array(image)
                            
                            prediction = model.predict(processed_image)
                            mask = prediction[0]
                            
                            if len(mask.shape) == 3:
                                mask = mask[:, :, 0]
                            
                            mask = (mask > 0.5).astype(np.uint8)
                            
                            st.header("Results")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.subheader("Segmentation Mask")
                                fig, ax = plt.subplots(figsize=(6, 6))
                                ax.imshow(mask, cmap='viridis')
                                ax.axis('off')
                                st.pyplot(fig)
                            
                            with col2:
                                st.subheader("Overlay Result")
                                overlay = create_overlay(original_array, mask)
                                st.image(overlay, caption="Segmentation Overlay", use_column_width=True)
                            
                            with col3:
                                st.subheader("Statistics")
                                mask_area = np.sum(mask)
                                total_area = mask.shape[0] * mask.shape[1]
                                coverage = (mask_area / total_area) * 100
                                
                                st.metric("Mask Coverage", f"{coverage:.2f}%")
                                st.metric("Mask Pixels", f"{mask_area:,}")
                                st.metric("Total Pixels", f"{total_area:,}")
                                st.metric("Image Size", f"{mask.shape[1]}x{mask.shape[0]}")
                                
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
    
    else:
        st.info("👆 Please upload your model file in the sidebar to get started")
        st.markdown("""
        ### Instructions:
        1. **Upload Model**: Use the sidebar to upload your `.h5` or `.keras` model file
        2. **Upload Image**: Upload an image you want to segment
        3. **Run Segmentation**: Click the button to see the results
        
        ### Supported Models:
        - U-Net
        - FCN
        - SegNet
        - Any TensorFlow/Keras segmentation model
        """)

if __name__ == "__main__":
    main()
