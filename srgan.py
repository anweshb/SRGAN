# %%
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, PReLU, BatchNormalization, LeakyReLU, Dense, add, Flatten, UpSampling2D, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
from sklearn.model_selection import train_test_split

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
def residual_block(inp):
    
    residual_block = Conv2D(64, (3,3), padding = 'same')(inp)
    residual_block = BatchNormalization(momentum = 0.5)(residual_block)
    residual_block = PReLU(shared_axes = [1,2])(residual_block)
    
    residual_block = Conv2D(64, (3,3), padding = 'same')(residual_block)
    residual_block = BatchNormalization(momentum = 0.5)(residual_block)
    
    return add([inp, residual_block])


# %%
def upscale_block(inp):
    
    upscale_block = Conv2D(256, (3,3), padding = 'same')(inp)
    upscale_block = UpSampling2D(size = 2)(upscale_block)
    upscale_block = PReLU(shared_axes = [1,2])(upscale_block)
    
    return upscale_block

# %%
def generator(inp, number_res_blocks = 16):
    
    gen = Conv2D(64, (9,9), padding = 'same')(inp)
    gen = PReLU(shared_axes = [1,2])(gen)
    
    temp = gen
    
    for _ in range(number_res_blocks):
        gen = residual_block(gen)
        
    gen = Conv2D(64, (3,3), padding = 'same')(gen)
    gen = BatchNormalization(momentum = 0.5)(gen)
    gen = add([gen,temp])
    
    gen = upscale_block(gen)
    gen = upscale_block(gen)
    
    output = Conv2D(3, (9,9), padding = 'same')(gen)
    
    return Model(inputs = inp, outputs = output)

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
def disc_block(inp, filters, bn = True, strides = 1):
    
    disc = Conv2D(filters, (3,3), padding = 'same', strides = strides)(inp)
    if bn:
        disc = BatchNormalization(momentum = 0.8)(disc)
    
    disc = LeakyReLU(alpha = 0.2)(disc)
    
    return disc

# %%
def create_discriminator(inp):

    filters = 64
    
    d1 = disc_block(inp, filters = filters, bn = False)
    d2 = disc_block(d1, filters = filters, strides = 2)
    d3 = disc_block(d2, filters = filters * 2)
    d4 = disc_block(d3, filters = filters * 2, strides = 2)
    d5 = disc_block(d4, filters = filters * 4)
    d6 = disc_block(d5, filters = filters * 4, strides = 2)
    d7 = disc_block(d6, filters = filters * 8)
    d8 = disc_block(d6, filters = filters * 8, strides = 2)
    
    d9 = Flatten()(d8)
    d10 = Dense(filters * 16)(d9)
    d11 = LeakyReLU()(d10)
    d12 = Dense(1, activation = 'sigmoid')(d11)
    
    return Model(inp, d12)

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
def build_vgg(hr_shape):
    
    vgg = VGG19(weights = 'imagenet', include_top = False, input_shape = hr_shape)
    
    return Model(inputs = vgg.inputs, outputs = vgg.layers[10].output)

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
##Sanity check

import random
import numpy as np
image_number = random.randint(0, len(lr_images)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(lr_images[image_number], (32, 32, 3)))
plt.subplot(122)
plt.imshow(np.reshape(hr_images[image_number], (128, 128, 3)))
plt.show()

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
lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

# %%
generator = generator(lr_ip)
generator._name = "GENERATOR"
generator.compile(loss = 'mse', optimizer = 'adam', metrics = ['loss'])
generator.summary()

# %%
discriminator = create_discriminator(hr_ip)
discriminator.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
discriminator._name = "DISCRIMINATOR"
discriminator.summary()
# %%
vgg = build_vgg((128,128,3))
print(vgg.summary())
vgg.trainable = False

# %%
def combined_model(generator, discriminator, vgg, lr_ip, hr_ip):
    
    gen_img = generator(lr_ip)
    
    ## Extracting VGG features of generated image
    gen_features = vgg(gen_img)
    real_features = vgg(hr_ip)
    
    discriminator.trainable = False
    
    return Model(inputs = [lr_ip, hr_ip], outputs = [real_features, gen_features])

srgan = combined_model(generator, discriminator, vgg, lr_ip, hr_ip)

# %%
srgan.compile(loss = ['binary_crossentropy','mse'], loss_weights = [1e-3, 1], optimizer = 'adam')
srgan.summary()

# %%
batch_size = 1

train_lr_batches = []
train_hr_batches = []

for batch in range(int(hr_train.shape[0] / batch_size)):
    start_idx = batch * batch_size
    end_idx = start_idx + batch_size
    train_lr_batches.append(lr_train[start_idx : end_idx])
    train_hr_batches.append(hr_train[start_idx : end_idx])

# %%
epochs = 5

for e in range(epochs):

    real_labels = np.ones((batch_size,1))
    fake_labels = np.zeros((batch_size,1))

    g_losses = []
    d_losses = []

    for batch in tqdm(range(len(train_hr_batches))):
        
        lr_imgs = train_lr_batches[batch]
        hr_imgs = train_hr_batches[batch]

        fake_imgs = generator.predict_on_batch(lr_imgs)

        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_labels)

        discriminator.trainable = False

        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

        image_features = vgg.predict(hr_imgs)
        fake_features = vgg.predict(fake_imgs)

        _, _, g_loss = srgan.train_on_batch([lr_imgs, hr_imgs], [fake_features, image_features])
        
        d_losses.append(d_loss)
        g_losses.append(g_loss)

        print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)

        if (e+1) % 1 == 0: #Change the frequency for model saving, if needed
        #Save the generator after every n epochs (Usually 10 epochs)
            generator.save("gen_e_"+ str(e+1) +".h5")






# %%


