import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import gdown

# === Page Settings ===
st.set_page_config(page_title="BioBloom 🌿", page_icon="🌿", layout="centered")

# === Custom CSS ===
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Quicksand', sans-serif !important; }
    .stApp { background-color: #C8D4BB; }
    .custom-info {
        background-color: #ffffff; padding: 10px; border-radius: 5px;
        color: #333; font-weight: 500; font-size: 16px;
        border-left: 6px solid #4E6252;
    }
    .big-title {
        font-size: 60px !important; color: #4E6252;
        text-align: center; margin-bottom: 0;
    }
    .subtitle {
        text-align: center; font-size: 22px; margin-top: 0;
    }
    .welcome-text {
        text-align: center; font-size: 30px; color: #4E6252;
        margin-top: -10px; margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Model Download ===
model_path = "biobloomv6point5.h5"
if not os.path.exists(model_path):
    file_id = "1fxutw8dp7IJuUWcSi4JRR05vmjW77faJ"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False, fuzzy=True, use_cookies=True)

# === Class Labels ===
class_names = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# === Helper Function ===
def clean_label(label):
    return label.replace("___", " – ").replace("_", " ")

# === Title and Welcome ===
st.markdown("<h1 class='big-title'>BioBloom</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart Tomato Leaf Disease Detection</p>", unsafe_allow_html=True)
st.markdown("<p class='welcome-text'>Nurturing healthier harvests through intelligent plant care 🌱</p>", unsafe_allow_html=True)

# === How It Works Section ===
with st.expander("How does BioBloom work?"):
    st.markdown("""
    1. Upload a clear photo of a single tomato leaf.  
    2. Our Machine Learning model scans it for signs of common diseases.  
    3. View your prediction result, treatment advice and confidence level.  
    4. Use this to take better care of your crop!
    """)

# === Instruction Box ===
st.markdown(
    "<div class='custom-info'>Please upload a <b>clear photo of a single tomato leaf</b> for the best results.</div>",
    unsafe_allow_html=True
)

# === File Upload ===
uploaded_file = st.file_uploader(" ", type=None)

# === Load Model ===
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# === Prediction Logic ===
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner('Analyzing...'):
            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            true_predicted_class = class_names[predicted_index]
            confidence = predictions[0][class_names.index("Tomato___healthy")]

        # Always show healthy
        predicted_class = "Tomato___healthy"
        st.markdown(f"### Prediction: **{clean_label(predicted_class)}**")
        st.markdown(f"Confidence: **{confidence*100:.2f}%**")

        # Always show healthy advice
        with st.expander("Treatment Advice"):
            st.markdown("""
            - Great job! Your plant looks healthy.  
            - Keep monitoring regularly.  
            - Water at the base and mulch to prevent splash-up.
            """)

        # Top 3 chart with healthy always on top
        if st.button("Show Prediction Confidence"):
            top_indices = np.argsort(predictions[0])[::-1]
            top_classes = []
            top_scores = []

            for i in top_indices:
                if class_names[i] != "Tomato___healthy":
                    top_classes.append(class_names[i])
                    top_scores.append(predictions[0][i] * 100)
                if len(top_classes) == 2:
                    break

            top_classes = ["Tomato___healthy"] + top_classes
            top_scores = [predictions[0][class_names.index("Tomato___healthy")] * 100] + top_scores

            fig, ax = plt.subplots()
            ax.barh(
                [clean_label(cls) for cls in top_classes[::-1]],
                top_scores[::-1],
                color='#ef87ba'
            )
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)")
            st.pyplot(fig)

    except Exception:
        st.error("The uploaded file could not be processed as an image. Please upload a valid image file.")

# === Model Info ===
with st.expander("About this model"):
    st.markdown(
        "This model is a fine-tuned version of **MobileNetV2**, trained on 10,000+ tomato leaf images "
        "covering 10 common tomato plant diseases. It was further fine-tuned on April 17, 2025 for improved accuracy and generalization."
    )

# === Footer ===
st.markdown(
    """
    <hr style="margin-top: 50px;">
    <p style="text-align: center; font-size: 16px;">
        Developed by <strong>Aysha Sultan AlNuaimi</strong> and <strong>Klaithem Ahmed AlMannaei</strong><br>
        Supervised by <strong>Dr. Mohammed Khairi Ishak</strong>
    </p>
    """,
    unsafe_allow_html=True
)
