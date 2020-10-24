#%%
from tensorflow import keras
from keras import layers
import tensorflow as tf
import PIL as pil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def img_to_array(img_path, resize = False, width = 256, height = 256) :
    img = pil.Image.open("7001-21.jpg")
    img = img.rotate(270)
    img = np.array(img,np.int32) / 255
    if(resize == True) : img = tf.image.resize(img,(width,height))
    return img
# %%

model = keras.models.Sequential()