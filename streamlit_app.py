import streamlit as st
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import warnings
from PIL import Image
import io

# disable warnings
warnings.filterwarnings('ignore')

model = load_model('trained_model.keras', custom_objects={'KerasLayer': hub.KerasLayer})

def predict(image):
    
    image_array = np.array(image)

    image = cv.cvtColor(image_array, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224), interpolation=cv.INTER_CUBIC)
    image = image.astype('float32')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    prediction = np.argmax(prediction)    
    label = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}
    return label[prediction]

def main():
    # title
    st.title("Flower Classification")

    # description
    st.text("Upload or take a picture of a flower for identification:\nThe flower will be classified as a daisy / dandelion / rose / sunflower / tulip")

    # image upload
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    prediction = None

    # predict
    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        resized_image = image.resize((224, 224))
        FRAME_WINDOW = st.image(resized_image)
        prediction = predict(image)
        st.success(f'The flower is a {prediction}.')
    else:
        st.info('Please upload an image or take a picture')


    # open camera
    '''
    run = st.checkbox('Open camera')
    FRAME_WINDOW = st.image([])
    camera = cv.VideoCapture(0)
    
    if not run:
        st.warning('Toggle the checkbox to open/close camera.')

    # take a picture
    if st.button('Take a picture'):
        _, frame = camera.read()
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        FRAME_WINDOW.image(image)
        run = False
        prediction = predict(image)
        st.success(f'The flower is a {prediction}.')
    '''

if __name__ == "__main__":
    main()
