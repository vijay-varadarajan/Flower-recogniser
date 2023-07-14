import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import warnings

# disable warnings
warnings.filterwarnings('ignore')

model = load_model('trained_model.keras', custom_objects={'KerasLayer': hub.KerasLayer})

def predict(path):
    image = cv.imread(path)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224), interpolation=cv.INTER_CUBIC)
    image = image.astype('float32')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)
    prediction = np.argmax(prediction)    
    label = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}
    print(label[prediction])

path = input('Enter the path of the image: ')
predict(path)