import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf
import streamlit as st


model = tf.keras.models.load_model("D:/Plant disease/plant_disease_model.keras")
class_indices = json.load(open("D:/Plant disease/class_indices.json"))

def load_preprocess_image(image_path, target_size = (224,224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis = 0)
    img_arr = img_arr.astype('float32') / 255.
    return img_arr


def predict_image(model, image_path, class_indices):
    preprocessed_img = load_preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction, axis = 1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

st.title('🌿 Plant disease classifier')

image = st.file_uploader('Upload an image\n', type = ['jpg', 'jpeg', 'png'])

if image is not None:
    img = Image.open(image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = img.resize((255,255))
        st.image(resized_img)
    

    with col2:
        if st.button('Classify'):
            prediction = predict_image(model, image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
