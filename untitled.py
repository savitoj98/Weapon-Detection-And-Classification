from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

x=[]
m,n = 240,240
model = tf.keras.models.load_model("Weapon-Detection-And-Classification/models/model_new.h5")
# gun_path = 'train/gun/'
# knife_path = 'train/knife/'
# gun_files=os.listdir(gun_path)
# knife_files = os.listdir(knife_path)

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(img)
        imrs = im.resize((m,n))
        imrs=img_to_array(imrs)/255
        imrs=imrs.transpose(2,0,1)
        imrs=imrs.reshape(3,m,n)
        dome = []
        dome.append(imrs)
        dome = np.array(dome)
        predictions = model.predict(dome)
        print (np.mean(predictions, axis = 0))

        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

show_webcam()
