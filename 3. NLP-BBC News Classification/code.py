#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import csv
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("bbc-text.csv")


# In[3]:


print(df)


# In[4]:


print(df.info())


# In[5]:



print(df.head(),"\n")
print(df.info())


# In[6]:


category = pd.get_dummies(df.category)
df_baru = pd.concat([df, category], axis=1)
df_baru = df_baru.drop(columns='category')
print(df_baru)


# In[7]:


article = df_baru['text'].values
label = df_baru[['business', 'entertainment', 'politics','sport','tech']].values
label = np.argmax(label, axis=1)


# In[8]:


from sklearn.model_selection import train_test_split
article_train, article_test, label_train, label_test = train_test_split(article, label, test_size=0.2)


# In[9]:


print(article.shape)


# In[10]:


print(label_train.shape)


# In[11]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(article_train) 
tokenizer.fit_on_texts(article_train)
 
sekuens_latih = tokenizer.texts_to_sequences(article_train)
sekuens_test = tokenizer.texts_to_sequences(article_test)
 
padded_latih = pad_sequences(sekuens_latih, maxlen=200, padding='post', truncating='post') 
padded_test = pad_sequences(sekuens_test, maxlen=200, padding='post', truncating='post') 


# In[12]:


print(padded_test.shape)


# In[14]:


label


# In[22]:


import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, LSTM, Flatten, Bidirectional
from tensorflow.keras.layers import Dropout, Embedding,GlobalMaxPool1D,SpatialDropout1D
from tensorflow.keras.models import Sequential
print(tf.__version__)
L2 = 0.0000001
embedding_dim=256
n_class = 5

clear_session()
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=embedding_dim))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(embedding_dim,kernel_initializer='orthogonal', kernel_regularizer=l2(L2), recurrent_regularizer=l2(L2),
         bias_regularizer=l2(L2))))
model.add(Dense(embedding_dim, activation='relu',kernel_regularizer=l2(L2), bias_regularizer=l2(L2)))
model.add(Dense(n_class, activation='softmax',kernel_regularizer=l2(L2), bias_regularizer=l2(L2) ))
model.summary()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.97):
            print("\nAkurasi telah mencapai > 97%!")
            self.model.stop_training = True

callbacks = myCallback()

model.compile(loss='sparse_categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])

num_epochs = 20
lstm = model.fit(padded_latih, label_train, epochs=num_epochs, 
                    validation_data=(padded_test, label_test),callbacks=[callbacks],verbose=2)


# In[23]:


plt.plot(lstm.history['accuracy'])
plt.plot(lstm.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.show()

plt.plot(lstm.history['loss'])
plt.plot(lstm.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.show()


# In[ ]:




