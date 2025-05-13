import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Set Streamlit page config
st.set_page_config(page_title="Tomato Leaf Health Checker", layout="centered")

# Title
st.title("🍅 Tomato Leaf Health Checker")
st.markdown("Upload a photo of your tomato leaf and let the AI *definitely not lie* to you about its health!")

# Load the trained model
@st.cache_resource
def load_plant_model():
    return load_model("model.h5")

model = load_plant_model()

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Utility to clean class labels
def clean_label(label):
    label = label.replace("Tomato___", "").replace("_", " ")
    return label.title()

# File uploader
uploaded_file = st.file_uploader("Upload an image of a tomato leaf 🍃", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        with st.spinner('Analyzing your *definitely healthy* tomato...'):
            predictions = model.predict(img_array)[0]

        # Get healthy class confidence
        healthy_index = class_names.index("Tomato___healthy")
        healthy_confidence = predictions[healthy_index]

        # 🍅 HILARIOUSLY FAKE VERDICT
        st.markdown("### 🧠 AI Verdict: **Tomato Healthy 🍅💪**")
        st.markdown(f"**Confidence Level:** `{healthy_confidence * 100:.2f}%` sure this leaf is killing it 🌿")
        st.info("Definitely not suspicious at all 😌")

        # Healthy treatment advice (always)
        with st.expander("🧪 Treatment Advice"):
            st.markdown("""
            - ✅ Great job! Your tomato plant looks healthy.  
            - 🕵️ Keep monitoring regularly just in case.  
            - 💧 Water at the base, not the leaves.  
            - 🌞 Ensure it gets 6–8 hours of sunlight.  
            - 🍃 Remove any yellow or damaged leaves if they appear.
            """)

        # Show confidence chart
        if st.button("Show Prediction Confidence"):
            st.subheader("🔬 Top 3 Prediction Confidence")

            # Sort predictions, exclude healthy, then take top 2
            top_indices = np.argsort(predictions)[::-1]
            top_disease_indices = [i for i in top_indices if i != healthy_index][:2]
            final_indices = [healthy_index] + top_disease_indices

            top_classes = [class_names[i] for i in final_indices]
            top_scores = [predictions[i] * 100 for i in final_indices]

            # Plot chart
            fig, ax = plt.subplots()
            ax.barh(
                [clean_label(cls) for cls in top_classes[::-1]],
                top_scores[::-1],
                color='#ef87ba'
            )
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"The uploaded file could not be processed as an image: {e}")
