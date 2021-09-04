#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL


# In[2]:


print('total Airplane images :', len(os.listdir('../../../Dataset/train/Airplane')))
print('total Bottle images :', len(os.listdir('../../../Dataset/train/Bottle')))
print('total Camera images :', len(os.listdir('../../../Dataset/train/Camera')))
print('total Mouse images :', len(os.listdir('../../../Dataset/train/Computer Mouse')))
print('total Cookie images :', len(os.listdir('../../../Dataset/train/Cookie')))
print('total Laptop images :', len(os.listdir('../../../Dataset/train/Laptop')))
print('total Phone images :', len(os.listdir('../../../Dataset/train/Mobile phone')))
print('total Motorcycle images :', len(os.listdir('../../../Dataset/train/Motorcycle')))
print('total Pool images :', len(os.listdir('../../../Dataset/train/Swimming pool')))
print('total Zebra images :', len(os.listdir('../../../Dataset/train/Zebra')))


# In[3]:


file_dir = '../../../Dataset/train/'


# In[4]:


import imageio
directory=os.listdir('../../../Dataset/train')
for each in directory:
    plt.figure()
    currentFolder = '../../../Dataset/train/' + each
    for i, file in enumerate(os.listdir(currentFolder)[5:9]):
        fullpath = currentFolder+ "/" + file
        print(fullpath)
        img=imageio.imread(fullpath)
        plt.title(each)
        plt.subplot(2, 2, i+1)
        plt.imshow(img)


# In[5]:


from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, UpSampling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.applications.xception import Xception
train_dir = os.path.join(file_dir)
train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=90,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode = 'nearest',
    validation_split=0.2) # set validation split


# In[12]:


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(220, 220),
    batch_size=16,
    class_mode='categorical',
    subset='training') # set as training data
validation_generator = train_datagen.flow_from_directory(
    train_dir, # same directory as training data
    target_size=(220, 220),
    batch_size=32,
    class_mode='categorical',
    subset='validation')


# In[27]:


filepath = 'best_model_openimage.h5'

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)
stop =  EarlyStopping(monitor='accuracy', min_delta=0.01, patience =4,
                      verbose=0, mode='auto', baseline=None, 
                      restore_best_weights=False)
callbacks = [checkpoint,stop]

resnet_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(220, 220, 3), classes = 10)

for layer in resnet_model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False
epochs=50
model = Sequential()
# model.add(UpSampling2D())
# model.add(UpSampling2D())
# model.add(UpSampling2D())
model.add(resnet_model)
model.add(GlobalMaxPooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(.25))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))


# In[28]:


model.compile(optimizer=tf.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])


# In[29]:


history = model.fit(train_generator,
                              validation_data=validation_generator,
                              epochs=epochs,
                              verbose=1, callbacks=callbacks)


# In[30]:


numOfEpoch = 21

plt.figure(figsize=(14, 4))

plt.subplot(1,2,1)
plt.plot(np.arange(0, numOfEpoch), history.history['loss'], 'bo', label='Training loss')
plt.plot(np.arange(0, numOfEpoch), history.history['val_loss'], 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(np.arange(0, numOfEpoch), history.history['accuracy'], 'bo', 
         label='Accuracy', c='orange')
plt.plot(np.arange(0, numOfEpoch), history.history['val_accuracy'], 'b', 
         label='Validation accuracy', c='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# In[ ]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# In[ ]:


with tf.io.gfile.GFile('model-cifar.tflite', 'wb') as f:
    f.write(tflite_model)


# In[ ]:




