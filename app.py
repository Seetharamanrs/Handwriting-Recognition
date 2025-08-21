import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load trained CNN model
path=r"D:\my_git\Handwriting-Recognition\Notebook\cnn_model.h5"
cnn_model = load_model(path)

st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) below and let the model predict it!")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",  # background black
    stroke_width=10,
    stroke_color="#FFFFFF",  # white ink
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Convert RGBA → grayscale
    img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # Resize to 28x28 (MNIST format)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # normalize
    img = img.reshape(1, 28, 28)

    # Predict
    prediction = cnn_model.predict(img)
    pred_digit = np.argmax(prediction)

    st.write(f"### 🧠 Prediction: {pred_digit}")
    st.bar_chart(prediction[0])
