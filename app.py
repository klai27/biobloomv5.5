import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import gdown

# === Page Settings ===
st.set_page_config(
    page_title="BioBloom 🌿",
    page_icon="🌿",
    layout="centered"
)

# === Custom CSS for Background ===
st.markdown(
    """
    <style>
    .stApp {
        background-color: #C8D4BB;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Download Model from Google Drive ===
model_path = "biobloomv6point5.h5"
if not os.path.exists(model_path):
    file_id = "1fxutw8dp7IJuUWcSi4JRR05vmjW77faJ"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False, fuzzy=True, use_cookies=True)

# === Verify Download ===
if os.path.exists(model_path):
    st.success(f"Model `{model_path}` downloaded successfully.")
    st.write("File size:", os.path.getsize(model_path), "bytes")
else:
    st.error("Model file not found. Please check the download link.")
    st.stop()

# === Class Names ===
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# === App Title ===
st.markdown(
    "<h1 style='text-align: center; color: #4E6252;'>BioBloom</h1>"
    "<p style='text-align: center; font-size: 18px;'>Smart Tomato Leaf Disease Detection</p>",
    unsafe_allow_html=True
)

# === Instructions ===
st.info("Please upload a **clear photo of a single tomato leaf** for the best results.")

# === Upload Image ===
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

# === Load Model ===
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# === Prediction ===
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

    # === Confidence Chart ===
    if st.button("Show Prediction Confidence"):
        st.subheader("Top 3 Prediction Confidence")
        top_indices = np.argsort(predictions[0])[::-1][:3]
        top_classes = [class_names[i] for i in top_indices]
        top_scores = [predictions[0][i] * 100 for i in top_indices]

        # Matching green palette
        colors = ['#4E6252', '#6B8E4E', '#A3B18A']

        fig, ax = plt.subplots()
        bars = ax.barh(top_classes[::-1], top_scores[::-1], color=colors[:len(top_classes)])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        st.pyplot(fig)

    # === Rerun ===
    if st.button("Try Another Image"):
        st.experimental_rerun()

# === Model Info ===
with st.expander("About this model"):
    st.markdown(
        "This model is a fine-tuned version of **MobileNetV2**, trained on 10,000+ tomato leaf images "
        "covering 10 common tomato plant diseases. It was further fine-tuned on April 17, 2025 for improved accuracy."
    )

# === Footer ===
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <p style="text-align: center; font-size: 16px;">
        Developed by <strong>Aysha Sultan AlNuaimi</strong> and <strong>Klaithem Ahmed AlMannaei</strong>
    </p>
    """,
    unsafe_allow_html=True
)
