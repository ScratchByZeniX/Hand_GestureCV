import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Human Gesture Recognition",
    page_icon="✋",
    layout="centered"
)

st.title("✋ Human Gesture Recognition")
st.write("Upload a hand gesture image to predict the gesture.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gesture_model.keras")

model = load_model()

# -----------------------------
# Load Labels
# -----------------------------
with open("labels.json", "r") as f:
    labels = json.load(f)

# Convert {label:index} to {index:label}
class_names = {v: k for k, v in labels.items()}

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose Gesture Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert Image
    img = np.array(image)

    # Resize
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img.astype("float32") / 255.0

    # Add Batch Dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    gesture = class_names[index]

    st.success(f"Predicted Gesture: {gesture}")

    st.info(f"Confidence: {confidence:.2f}%")