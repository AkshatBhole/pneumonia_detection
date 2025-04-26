import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('trained.h5')  # Ensure this path is correct

st.title("🩺 Pneumonia Detection from Chest X-ray")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # ✅ Resize to match model's input shape (e.g., 300x300)
    image = image.resize((300, 300))  # Width x Height

    # ✅ Convert to numpy array and normalize
    image_array = np.array(image) / 255.0  # Shape: (300, 300, 3)

    # ✅ Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 300, 300, 3)

    # ✅ Predict
    prediction = model.predict(image_array)
    st.write("Raw prediction value:", prediction)

    result = "Pneumonia Detected ❗" if prediction[0][0] > 0.5 else "Normal ✅"
    st.subheader("Prediction Result:")
    st.success(result)
