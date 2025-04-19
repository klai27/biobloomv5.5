
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from PIL import Image
import os
import subprocess

# === Step 1: Download Model from Google Drive if not exists ===
model_path = "mobilenetv2_finetuned_2025-04-17.h5"
if not os.path.exists(model_path):
    file_id = "1nqZLHbc0fMtl1bBXWx-84XHjzXDnipOE"
    subprocess.run(["pip", "install", "gdown"])
    subprocess.run(["gdown", f"https://drive.google.com/uc?id={file_id}", "-O", model_path])

# === Step 2: Load Class Indices ===
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# === App Title ===
st.markdown(
    "<h1 style='text-align: center; color: #4E6252;'>BioBloom</h1>"
    "<p style='text-align: center; font-size: 18px;'>Smart Tomato Leaf Disease Detection</p>",
    unsafe_allow_html=True
)

# === Instructions ===
st.info("Please upload a **clear photo of a single tomato leaf** for the best results.")

# === File Upload ===
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

# === Load Model ===
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("Model loaded and ready.")
else:
    st.error("Model file not found. Please upload the model.")
    st.stop()

# === Image Prediction ===
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner('Analyzing...'):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index]

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"Confidence: **{confidence*100:.2f}%**")

    # === Confidence Bar Chart ===
    if st.button("Show Prediction Confidence"):
        st.subheader("Top 3 Prediction Confidence")
        top_indices = np.argsort(predictions[0])[::-1][:3]
        top_classes = [class_names[i] for i in top_indices]
        top_scores = [predictions[0][i] * 100 for i in top_indices]

        fig, ax = plt.subplots()
        ax.barh(top_classes[::-1], top_scores[::-1], color='green')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        st.pyplot(fig)

    # === Try Another Image ===
    if st.button("Try Another Image"):
        st.experimental_rerun()

# === Info About the Model ===
with st.expander("About this model"):
    st.markdown(
        "This model is a fine-tuned version of **MobileNetV2**, trained on 10,000+ tomato leaf images "
        "covering 10 common tomato plant diseases. It was further fine-tuned on April 17, 2025 for better accuracy and generalization."
    )
