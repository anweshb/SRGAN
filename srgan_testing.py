# %%
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, PReLU, BatchNormalization, LeakyReLU, Dense, add, Flatten, UpSampling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
from sklearn.model_selection import train_test_split
from numpy.random import randint


# %% [markdown]
# ![image.png](attachment:image.png)

# %%
n = 25000

lr_list = os.listdir("lr_images")[:n]

try:
    lr_list.remove(".DS_Store")
except ValueError:
    pass

lr_images = []

for img in lr_list:
    img_lr = cv2.imread("lr_images/"+img)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    lr_images.append(img_lr)

hr_list = os.listdir("hr_images")[:n]

try:
    hr_list.remove(".DS_Store")    
except ValueError:
    pass

hr_images = []

for img in hr_list:
    img_hr = cv2.imread("hr_images/"+img)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    hr_images.append(img_hr)


# %%
lr_images = np.array(lr_images)
hr_images = np.array(hr_images)

# %%
lr_images = lr_images / 255.
hr_images = hr_images / 255.

# %%
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, 
                                                      test_size=0.3, random_state=42)

# %%
hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])


# %%
generator = load_model('gen_e_5.h5', compile=False)

[X1, X2] = [lr_test, hr_test]
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# generate image from source
gen_image = generator.predict(src_image)

