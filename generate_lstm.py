
# coding: utf-8

# In[3]:

import numpy as np
import theano as theano
import theano.tensor as T
import time
import operator
from utils import load_data, load_model_parameters_theano, generate_sentences
#from gru_theano import *
from lstm_theano import *
import sys


# In[4]:

# Load data (this may take a few minutes)
VOCABULARY_SIZE = 8000
X_train, y_train, word_to_index, index_to_word = load_data("data/lyrics.txt", VOCABULARY_SIZE)


# In[21]:

# Load parameters of pre-trained model
model = load_model_parameters_theano('./data/LSTM-2016-04-12-05-40-8000-48-128.dat.npz')


# In[2]:

# Build your own model (not recommended unless you have a lot of time!)

LEARNING_RATE = 1e-3
NEPOCH = 20
HIDDEN_DIM = 128

 #model = LSTMTheano(VOCABULARY_SIZE, HIDDEN_DIM)

 #t1 = time.time()
 #model.sgd_step(X_train[0], y_train[0], LEARNING_RATE)
 #t2 = time.time()
 #print "SGD Step time: ~%f milliseconds" % ((t2 - t1) * 1000.)

 #train_with_sgd(model, X_train, y_train, LEARNING_RATE, NEPOCH, decay=0.9)


# In[23]:

#generate_sentences(model, 10, index_to_word, word_to_index)
generate_sentences(model, index_to_word, word_to_index)

# In[ ]:



