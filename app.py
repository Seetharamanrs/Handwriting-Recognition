import os
import numpy as np
import cv2
import joblib
import sklearn
import streamlit as st
from streamlit_drawable_canvas import st_canvas

model_path = r"D:\my_git\Handwriting-Recognition\Notebook\rf_model.joblib"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    model = joblib.load(model_path)


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
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

# if st.button('Predict'):
    
#     test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     val = model.predict(test_x.reshape(1, 28, 28))
#     st.write(f'result: {np.argmax(val[0])}')
#     st.bar_chart(val[0])
if st.button('Predict'):
    if canvas_result.image_data is None:
        st.warning("Draw a digit first!")
    else:
        # Convert to grayscale
        test_x = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_BGR2GRAY)
        test_x_resized=cv2.resize(test_x,(28,28),interpolation=cv2.INTER_NEAREST)
        test_x_flat = test_x_resized.reshape(1, -1).astype(int)
        val = model.predict_proba(test_x_flat)
        pred = model.predict(test_x_flat)
        st.write(f"Result: {pred[0]}")
        st.bar_chart(val[0])
