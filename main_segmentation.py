import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.title("🖼️ Image Segmentation")

@st.cache_resource
def load_model():
    try:
        model_path = "/content/drive/MyDrive/segmentation_model.h5"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

model = load_model()

if model:
    st.success("Model loaded!")
    uploaded_file = st.file_uploader("Upload image", type=['jpg', 'png'])
    
    if uploaded_file and st.button("Segment"):
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
else:
    st.error("Model not found!")
