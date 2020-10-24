#%%
import tensorflow as tf
import PIL as pil
from PIL import ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers,Sequential
import os

# %%
#data preprocessing (remove the suroundding noises and emphasize the crack)
def img_to_array(img_path, augmented=False) :
    width = 128
    height = 128
    img = pil.Image.open(img_path)
    img = (np.array(img,np.int32) + 120) / 255
    img = tf.image.adjust_contrast(img,3)
    #img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img,(width,height))
    if(augmented) : 
        arr = np.empty((0,width,height,1),np.int32)
        arr = np.append(arr,img)
        for i in range(3) :
            img = tf.image.rot90(img)
            arr = np.append(arr,img)
        arr = arr.reshape((4,width,height,3))
        return arr
    return img

# %%
img = img_to_array("7004-142.jpg",augmented=True)
for i in range(img.shape[0]) :
    plt.imshow(img[i])
    plt.show()

#%%
#data preparation loop
raw_data_path = "D:\\Python projects\\hackathon dataset\\Decks\\Cracked\\";
processed_data_path = "D:\\Python projects\\hackathon dataset\\Processed\\Cracked\\"

for file in os.listdir(raw_data_path):
    pathname = raw_data_path + file
    img = img_to_array(pathname, augmented=True)
    for i in range(img.shape[0]) :
        name = file.split(".")[0]
        extension = pathname.split(".")[1]
        full_path = processed_data_path + name + "-" + str(i) + "." + extension
        plt.imshow(img[i])
        plt.axis('off')
        plt.savefig(full_path)

#%%
img = pil.Image.open("7001-2-0.jpg")
img = np.array(img,np.int32) / 255
img = tf.image.resize(img,(128,128))
print(img.shape)

# %%
#build a model
model = Sequential([
    layers.Conv2D(64,(3,3),activation="relu",input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax") #binary output

model.compile(
    optimizer="adam", 
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# %%
