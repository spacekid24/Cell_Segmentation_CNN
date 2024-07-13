#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import h5py
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
width = 256
height = 256
channels = 1
train_path = r"./Data/Vertical_Images"
train_ids = 1774
X_train = np.zeros((train_ids, height, width), dtype = np.uint8)
Y_train = np.zeros((train_ids, height, width), dtype = bool)
folders = os.listdir(train_path)
for folder in folders:
    subfolder = os.path.join(train_path, folder)
    if folder == "Original":
        for img, n in zip(os.listdir(subfolder), range(train_ids)):
            img_path = os.path.join(subfolder, img)
            image = plt.imread(img_path)
            X_train[n] = image
    if folder == "Labeled":
        for img, n in zip(os.listdir(subfolder), range(train_ids)):
            img_path = os.path.join(subfolder, img)
            img_mask = plt.imread(img_path)
            Y_train[n] = img_mask         
X_train = np.expand_dims(X_train, axis=-1)
Y_train = np.expand_dims(Y_train, axis=-1)
print("X_train and Y_train ready")
print(X_train.shape)
print(Y_train.shape)
print("Model being defined")
inputs = tf.keras.layers.Input((height, width, channels))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)
c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
c2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
c3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
c4 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
c5 = tf.keras.layers.Conv2D(1024, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(1024, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
t6 = tf.keras.layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
t6 = tf.keras.layers.concatenate([t6,c4])
c6 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(t6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
t7 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
t7 = tf.keras.layers.concatenate([t7,c3])
c7 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(t7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
t8 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
t8 = tf.keras.layers.concatenate([t8,c2])
c8 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(t8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
t9 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
t9 = tf.keras.layers.concatenate([t9,c1])
c9 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(t9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

#model ouput
outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
print("Setting Callbacks")
early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs')
print("Model being trained")
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=32, epochs=100, callbacks=[early_stopping, tensorboard], shuffle=True)
model.save_weights("weights1774.weights.h5")