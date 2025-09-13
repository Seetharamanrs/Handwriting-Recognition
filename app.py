import os
import numpy as np
import cv2
import joblib
import tensorflow as tf
import sklearn
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

@st.cache_resource
def load_cnn_model(weights_path):
    def build_model():
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        return model

    model = build_model()
    model.load_weights(weights_path)
    return model


os.makedirs("Notebook", exist_ok=True)
model_path = r"Notebook/cnn_model.weights.h5"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
     model = load_cnn_model(model_path)


# st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title('My Digit Recognizer')
st.markdown('''
Try to write a digit!
''')

# data = np.random.rand(28,28)
# img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)

SIZE = 200
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=10,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)
    img_normalized = img_resized.astype('float32') / 255.0
    img_reshaped = np.expand_dims(img_normalized, axis=-1)
    model_input = np.expand_dims(img_reshaped, axis=0)
    st.write("Model Input")
    st.image(cv2.resize(img_resized, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST))
# if st.button('Predict'):
    
#     test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     val = model.predict(test_x.reshape(1, 28, 28))
#     st.write(f'result: {np.argmax(val[0])}')
#     st.bar_chart(val[0])
if st.button("Predict"):
    if canvas_result.image_data is None:
        st.warning("Draw a digit first!")
    else: 
        test_x = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_BGR2GRAY)

        test_x_resized = cv2.resize(test_x, (28,28), interpolation=cv2.INTER_AREA)
        if np.mean(test_x_resized) > 127:
            test_x_resized = 255 - test_x_resized
        _, test_x_resized = cv2.threshold(test_x_resized, 127, 255, cv2.THRESH_BINARY)

        test_x_normalized = test_x_resized.astype("float32") / 255.0

        model_input = np.expand_dims(np.expand_dims(test_x_normalized, axis=-1), axis=0)
        val = model.predict(model_input)              
        pred_class = np.argmax(val, axis=1)[0]        
        st.write(f"Result: {pred_class}")
        st.bar_chart(val[0])         
