import tensorflow as tf
import PIL as pil
import numpy as np
import pandas as pd
from tensorflow.keras import models

def img_to_array(img_path, augmented=False) :
    width = 128
    height = 128
    img = pil.Image.open(img_path)
    img = (np.array(img,np.int32) + 120) / 255
    img = tf.image.adjust_contrast(img,3)
    #img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img,(width,height))
    arr = []
    arr.append(img)
    if(augmented) : 
        #arr = np.empty((0,width,height,1),np.int32)
        #arr = np.append(arr,img)
        for i in range(3) :
            img = tf.image.rot90(img)
            arr.append(img)
        #arr = arr.reshape((4,width,height,3))
    return arr

def predict_custom_img(img_path : str) : #use to predict custom images
    img = img_to_array(img_path)
    img = np.array(img).reshape((1,128,128,3))
    model = models.load_model("./finalModel.h5")
    prediction = model.predict(img)[0][0]
    type = ""
    if(prediction >= 0.7) : type = "This image contains cracks"
    else :  type = "This image has no defect"
    return type