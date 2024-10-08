#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:55:37 2024

@author: catarinalopesdias
"""

from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np    



class CreateMaskLayer(Layer):


    def __init__(self):
        super().__init__()

    #function from stefan heber
    
    def apply_random_brain_mask(self, data):
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
        

        
        
    def call(self, input_tensor): 
        print(" ===== start mask layer ======")
        data = input_tensor
        print("initial data shape", data.shape)
        data =  data[0,:,:,:,0] # IF 4 D        
        print("data shape before function", data.shape)
        mask, dataMask = self.apply_random_brain_mask( data)
        
        
        dataMask = tf.expand_dims(dataMask, 0)
        dataMask = tf.expand_dims(dataMask, 4)

        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, 4)

        print("=== end mask layer =================")
        
        return mask, dataMask

        

        
