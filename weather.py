import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import joblib


# Load the model and label encoder
model = load_model('best_weather_model.h5')
label_encoder = joblib.load('label_encoder.pkl')

st.title("Weather Image Classifier 🌤️⛈️❄️")
st.write("Upload a weather image (64x64) to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((64, 64))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Prediction:** {predicted_label}")
