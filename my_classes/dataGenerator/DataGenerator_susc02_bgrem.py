#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:58:49 2024

@author: catarinalopesdias
"""

import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt


class DataGeneratorUniform(keras.utils.Sequence):
    'Generates data for Keras'


    def __init__(self, list_IDs, # labels,
                 batch_size=1, dim=(128,128,128), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    
    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
      # Generate data
      X, Y = self.__data_generation(list_IDs_temp)
    
      return X, Y
    
    
    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
      
    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      Y = np.empty((self.batch_size, *self.dim, self.n_channels))
      #Z = np.empty((self.batch_size, *self.dim, self.n_channels))

      #print(list_IDs_temp)
      # Generate data
      
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
         
          loaded = np.load('datasynthetic/uniform02/npz/' + ID + '.npz')

          loaded =loaded['arr_0']


          loaded = np.expand_dims(loaded, 4)


           # Store class
           #y[i] = self.labels[ID]
          X[i,:] = loaded[2,:] #phase+bg
          Y[i,:] = loaded[1,:] #phase

          
          #bla = loaded[1,:][64,:,:]
          #if i < 5:
           #   plt.imshow(Y[i,:,:,64,0], cmap='gray',  vmin=-0.4, vmax=0.4) 
            #  plt.show()  

      return X, Y#[Y,Z]




