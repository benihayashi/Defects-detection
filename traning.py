#%%
import tensorflow as tf
import PIL as pil
from PIL import ImageOps
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from tensorflow.keras import models,layers,Sequential, callbacks, losses
import os
import random

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
    arr = []
    arr.append(img)
    if(augmented) : 
        #arr = np.empty((0,width,height,1),np.int32)
        #arr = np.append(arr,img)
        #arr = []
        arr.append(img)
        for i in range(3) :
            img = tf.image.rot90(img)
            arr.append(img)
        #arr = arr.reshape((4,width,height,3))
    return arr
    #return img

# %%
# img = img_to_array("7004-142.jpg",augmented=True)
# for i in range(img.shape[0]) :
#     plt.imshow(img[i])
#     plt.show()

#%%
#data preparation loop
# raw_data_path = "D:\\Python projects\\hackathon dataset\\Decks\\Cracked\\";
# processed_data_path = "D:\\Python projects\\hackathon dataset\\Processed\\Cracked\\"

# for file in os.listdir(raw_data_path):
#     pathname = raw_data_path + file
#     img = img_to_array(pathname, augmented=True)
#     for i in range(img.shape[0]) :
#         name = file.split(".")[0]
#         extension = pathname.split(".")[1]
#         full_path = processed_data_path + name + "-" + str(i) + "." + extension
#         plt.imshow(img[i])
#         plt.axis('off')
#         plt.savefig(full_path)

#%%
def batch_preparation() :
    cracked_path = "../../Decks/Cracked/"
    not_cracked_path = "../../Decks/Non-Cracked/"

    cracked = os.listdir(cracked_path) 
    cracked_len = len(cracked) - 1
    not_cracked = os.listdir(not_cracked_path)
    not_cracked_len = len(not_cracked) - 1

    images_dict = {}

    cracked_num = random.randint(0,2000)
    not_cracked_num = 2000 - cracked_num

    #load in the files in cracked deck file
    for i in range(cracked_num) :
        file = cracked[random.randint(0,cracked_len)];
        img = img_to_array(cracked_path + file, augmented=False)
        for i in img:
            images_dict.update({i.ref():1})

    for i in range(not_cracked_num) :
        file = not_cracked[random.randint(0,not_cracked_len)];
        img = img_to_array(not_cracked_path + file, augmented=False)
        for i in img:
            images_dict.update({i.ref():0})
    
    return images_dict

#%%
imgs = batch_preparation()

cols = ["isCracked"]
train_data = pd.DataFrame.from_dict(imgs,orient='index', columns=cols)

#shuffle the data
#n = num of row we want, replace = false, we dont want same data in next sample
sampled = train_data.sample(n=2000,random_state = 1,replace = False)
#print(train_data)

derefered_images = []
#split the data into images and labels
train_images = np.array(sampled.index.values.tolist()) #<- list of list

for tensor in train_images:
    derefered_images.append(np.array(tensor.deref()))

#print(np.asarray(derefered_images).shape)
train_images = np.array(derefered_images)
train_labels = np.array(sampled['isCracked'].tolist())

#train_labels = train_labels.tolist()
#shuffle the data
#split the data into images and labels
#feed the data 
#%%

# %%
#build a model
model = Sequential([
    layers.Conv2D(128,(3,3),activation="relu",input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256,(5,5),activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256,(5,5),activation="relu"),
    layers.Flatten(),
])


# %%
def testingAccuracy():
    crackedAccuList = []
    notCrackedAccuList = []
    cracked_path = "../../Decks/Cracked/"
    not_cracked_path = "../../Decks/Non-Cracked/"
    cracked = os.listdir(cracked_path) 
    not_cracked = os.listdir(not_cracked_path)
    model = models.load_model("finalModel.h5")
    for i in range(0,len(cracked),20):
        testImage = img_to_array(cracked_path + cracked[i])
        testImage = np.array(testImage).reshape(1,128,128,3)
        prediction = model.predict(testImage)
        crackedAccuList.append(prediction[0][0])
        print("Cracked Prediction {}, accuracy: {}".format(i,prediction[0]))
    for i in range(0,len(not_cracked),150):
        testImage = img_to_array(not_cracked_path + not_cracked[i])
        testImage = np.array(testImage).reshape(1,128,128,3)
        prediction = model.predict(testImage)
        notCrackedAccuList.append(prediction[0][0])
        print("Not Cracked Prediction {}, accuracy: {}".format(i,prediction[0]))
    crackedAccu = np.array(crackedAccuList).mean()
    notCrackedAccu = np.array(notCrackedAccuList).mean()
    print("Average prediction for cracked: {}".format(crackedAccu))
    print("Average prediction for non-cracked: {}".format(notCrackedAccu))

testingAccuracy()
