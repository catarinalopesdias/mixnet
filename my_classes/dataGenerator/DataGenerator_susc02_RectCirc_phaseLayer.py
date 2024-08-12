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


class DataGeneratorUniform_RecCirc_phase(keras.utils.Sequence):
    'Generates data for Keras'

    def __apply_random_brain_mask(self, data):
        """Apply a random brain mask to the data (tensorflow graph update)

        Parameters
        ----------
        data : tf tensor
            Input data of size HxWxD

        Returns
        -------
        tf tensor
            Result of size HxWxD
        tf tensor
            Mask of size HxWxD
        """
        
        shape = [128,128,128]#data.get_shape().as_list()
        ndim = len(shape)
        
        # number of gaussians to be used
        num_gaussians = 20

        # sigma range in terms of percentage wrt the dim
        sigma_range = [0.15, 0.2]

        # init volume
        
        print("shape mask0", shape)
        mask0 = tf.zeros(shape, dtype=tf.bool)
        

        i0 = tf.constant(0)

        def _cond(i, mask):
            return tf.less(i, num_gaussians)

        def _body(i, mask):

            # create random positions inside dim range (clustered in the center)
            mu = tf.random.normal([ndim],
                                  mean=0.5,
                                  stddev=0.075)#,
                                  #seed=self.para['seed'] *
                                  #np.random.randint(1000))
            mu = tf.multiply(mu, shape)
            # create random sigma values
            sigma = tf.stack([
                sigma_range[0] * shape[i] + tf.random.uniform(
                    [])#, seed=self.para['seed'] * np.random.randint(1000)) 
                *
                (sigma_range[1] - sigma_range[0]) * shape[i]
                for i in range(ndim)
            ])

            #print(sigma)
            gauss_function = self.__calc_gauss_function(mu=mu,
                                                        sigma=sigma,
                                                        dim=shape)


            # update mask
            mask = tf.logical_or(gauss_function > 0.5, mask)

            i += 1
            return (i, mask)

        i, mask = tf.while_loop(_cond,
                                _body,
                                loop_vars=(i0, mask0),
                                parallel_iterations=1,
                                name="brain_mask_loop")


#        return tf.multiply(tf.to_float(mask), data), mask
        mask = tf.cast(mask, tf.float32)      
        
        masked_data = tf.multiply(mask, data)


        return mask, masked_data


    
    
    def __calc_gauss_function(self, mu, sigma, dim):
        """Calculates a gaussian function (update tensorflow graph).

        Parameters
        ----------
        mu: tuple
            Predefined center of the pdf (x,y,z)
        sigma: float
            Parameter
        dim: tuple
            Dimension of the output (width, height, depth)

        Returns
        -------
        tf tensor
            The desired gaussian function (max value is 1)
        """
        ndim = len(dim)

        linspace = [
            tf.linspace(0.0, dim[i] - 1.0, dim[i]) for i in range(ndim)
        ]
        coord = tf.meshgrid(*linspace, indexing='ij')

        # center at given point
        coord = [coord[i] - mu[i] for i in range(ndim)]

        # create a matrix of coordinates
        coord_col = tf.stack([tf.reshape(coord[i], [-1]) for i in range(ndim)],
                             axis=1)

        covariance_matrix = tf.linalg.diag(tf.square(sigma))

        z = tf.matmul(coord_col, tf.linalg.inv(covariance_matrix))
        z = tf.reduce_sum(tf.multiply(z, coord_col), axis=1)
        z = tf.exp(-0.5 * z)

        # we dont use the scaling
        #z /= np.sqrt(
        #    (2.0 * np.pi)**len(dim) * np.linalg.det(covariance_matrix))

        # reshape to desired size
        z = tf.reshape(z, dim)

        return z
    



    def __init__(self, list_IDs, # labels,
                 batch_size=1, dim=(128,128,128), n_channels=1, shuffle=False):#shuffle - false
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        #self.maxsize = (dim[2],)
    
    
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
      #X, Y = self.__data_generation(list_IDs_temp)
      #[X1,X2,X3], Y = self.__data_generation(list_IDs_temp)
      [X1,X2], Y = self.__data_generation(list_IDs_temp)

    
      #return [X1,X2,X3], Y
      return [X1,X2], Y

    
    
    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
      
    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      mask = np.empty((self.batch_size, *self.dim, self.n_channels))
      dataMask = np.empty((self.batch_size, *self.dim, self.n_channels))

      X1 = np.empty((self.batch_size, *self.dim, self.n_channels))
      X2 = np.empty((self.batch_size, *self.dim, self.n_channels))
      Y = np.empty((self.batch_size, *self.dim, self.n_channels))
      #X3 = np.empty((self.batch_size, *self.maxsize))

      #Z = np.empty((self.batch_size, *self.dim, self.n_channels))

      #print(list_IDs_temp)
      # Generate data
      
      for i, ID in enumerate(list_IDs_temp):
          # Store sample
          #print("i",i)
          #print("id",ID)
         
            
          loaded = np.load('datasynthetic/uniform02RectCircle-Mask-MaskedPhase-Phase/training/' + ID + '.npz')
          #loaded = np.load('datasynthetic/gt1bg500_normal01evenlessbgnoartifacts/npz/' + ID + '.npz')


          loaded =loaded['arr_0']
          print("loaded shape")
          print(loaded.shape)
          loaded = np.expand_dims(loaded, 4)
          print("expanded")
          print(loaded.shape)

           # Store classII
           #y[i] = self.labels[ID]
          #X1[i,:] = loaded[0,:] #mask
          X2[i,:] = loaded[2,:] #phase
          phase = loaded[2,:] #phase
          print("phase")
          print(phase.shape)
          
          mask, dataMask = self.__apply_random_brain_mask( phase[:,:,:,0])
          print("mask shape")
          print(mask.shape)
          print("datamask shape")
          print(dataMask.shape)
          
          mask = np.expand_dims(mask, 3)
          dataMask = np.expand_dims(dataMask, 3)
          print("mask shape extended ")
          print(mask.shape)
          print("datamask shape extended")
          print(dataMask.shape)

          
          X1[i,:] = mask
          X2[i,:]  = dataMask
          Y[i,:]  = dataMask

          
          #if i < 5:
           #   plt.imshow(Y[i,:,:,64,0], cmap='gray',  vmin=-0.4, vmax=0.4) 
            #  plt.show()  

      return [X1,X2], Y  #X,Y Y#




