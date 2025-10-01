import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Image Segmentation", layout="wide")

st.title("Deep Learning Image Segmentation")
st.write("Upload an image for semantic segmentation")

@st.cache_resource
def load_segmentation_model():
    try:
        model_path = "/content/drive/MyDrive/segmentation_unet_final.h5"
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, custom_objects={
                'dice_coef': dice_coef,
                'iou_coef': iou_coef,
                'f1_score': f1_score
            })
            st.success("✅ Model loaded successfully from Google Drive!")
            return model
        else:
            st.error(f"❌ Model not found at: {model_path}")
            st.info("📁Please make sure the model file exists in your Google Drive")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    if len(image_array.shape) == 2:  
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:  
        image_array = image_array[:, :, :3]
    return np.expand_dims(image_array, axis=0)

def create_overlay(original, mask, alpha=0.5):
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]))
    mask_colored = np.zeros_like(original)
    mask_colored[:, :, 1] = mask_resized * 255 
    
    overlay = cv2.addWeighted(original, 1 - alpha, mask_colored, alpha, 0)
    return overlay

def main():
    model = load_segmentation_model()
    
    if model is None:
        st.error("Failed to load segmentation model. Please check the model path.")
        return
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png','tif','tiff','TIF','TIFF'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        processed_image = preprocess_image(image)
        original_array = np.array(image)
        
        if st.button("Run Segmentation"):
            with st.spinner("Processing image..."):
                try:
                    prediction = model.predict(processed_image)
                    mask = prediction[0]  
                    
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]  
                    
                    mask = (mask > 0.5).astype(np.uint8)
                    
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
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

if __name__ == "__main__":

    main()








