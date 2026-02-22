import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os


# So streamlit_drawable_canvas is a library that provides a drawable canvas component for Streamlit applications. 
# It allows users to draw on a canvas using various tools such as freehand drawing, rectangles, circles, and more. 
# The drawn content can be captured and processed within the Streamlit app, making it useful for applications 
# like image annotation, sketching, or any interactive drawing needs.



def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "model", "model.keras")
    return tf.keras.models.load_model(model_path)

def onClickButton(canvas_result,col2):
    with col2:
        model=load_model()
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            img = Image.fromarray((img[:, :, 0]).astype("uint8"))
            img = img.resize((28, 28))
            img_array = np.array(img)
            img_array = 255 - img_array
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, 28 * 28)

            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)

            st.subheader(f"Predicted Digit: {predicted_digit}")
            st.bar_chart(prediction[0])

def main():
    st.set_page_config(
        page_title="Handwritten Digit Recognition",
        page_icon=":numbers:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    with st.container():
        st.markdown(
            """
            <div style='text-align: center;'>
                <h1 style='margin-bottom: 0.5em;'>Handwritten Digit Recognition</h1>
                <p style='font-size: 1.2em;'>Draw a digit (0–9) below and click Predict</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    col0, col1, col2, col3 = st.columns([1,3,3,1])
    with col1:
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=15,
            stroke_color="black",
            background_color="white",
            height=480,
            width=480,
            drawing_mode="freedraw",
            key="canvas1",
        )
        if st.button("Predict"):
            onClickButton(canvas_result,col2)



if __name__ == "__main__":
    main()