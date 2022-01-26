#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.utils import np_utils


# In[2]:


df = pd.read_csv('real_309_4_en1.csv', header = 0)

df = df.drop(['law','mind', 'support'], axis =1)


# In[3]:


df.columns


# In[4]:


dataset = df.values
X = dataset[:,0:11].astype(float)
y = dataset[:,11]


# In[5]:


seed = 7
numpy.random.seed(seed)


# In[6]:


encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[7]:


dummy_y


# In[8]:


def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=11, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[9]:


estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)


# In[10]:


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
estimator.fit(X_train, y_train)


# In[12]:


pred = estimator.predict(X_test)

init_lables = encoder.inverse_transform(pred)

seed = 42
np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)


# In[ ]:


print(results)


# In[ ]:




