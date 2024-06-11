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

from tensorflow.experimental import numpy as tnp # start using tnp instead of numpy or math library#test tnp.pi, tnp.e
from backgroundfieldandeffects.generate_backgroundfield_steffen_function import add_z_gradient_SMALL
    


class CreateMaskLayer(Layer):
    
    def __init__(self):
        super().__init__()
        self.grad

        
    #def build(self, input_shape):   
        
        
    def call(self, inputs):
        #num_rows, num_columns = tf.shape(inputs)[1], tf.shape(inputs)[2]

        # make sense of inputs   
        
        
        mask =  inputs[0]
        mask =  mask[0,:,:,:,0] # IF 4 D

        shape = data.get_shape().as_list()
        ndim = len(shape)



        # number of gaussians to be used
        num_gaussians = 20

        # sigma range in terms of percentage wrt the dim
        sigma_range = [0.15, 0.2]       
        #########################################
        for i in range(num_gaussians):

            # create random positions inside dim range (clustered in the center)
            #mu = np.random.normal(0.5, 0.075, len(dim)) * dim

            # create random positions inside dim range (clustered in the center)
            mu = tf.random.normal([ndim],
                                  mean=0.5,
                                  stddev=0.075)
            mu = tf.multiply(mu, shape)



            # create random sigma values
            #sigma = np.array([
            #    sigma_range[0] * dim[i] + np.random.rand() *
            #    (sigma_range[1] - sigma_range[0]) * dim[i]
            #    for i in range(len(dim))
            #])

            # create random sigma values
            sigma = tf.stack([
                sigma_range[0] * shape[i] + tf.random.uniform(
                    []) *
                (sigma_range[1] - sigma_range[0]) * shape[i]
                for i in range(ndim)
            ])


            gauss_function = calc_gauss_function_np(mu=mu,
                                                            sigma=sigma,
                                                            dim=dim)

            # update mask
            mask = np.logical_or(gauss_function > 0.5, mask)

        return np.multiply(mask, data), mask
 

        
    
def apply_random_brain_mask_tf( data):

        # init volume
        mask0 = tf.zeros(shape, dtype=tf.bool)

        i0 = tf.constant(0)

        def _cond(i, mask):
            return tf.less(i, num_gaussians)

        def _body(i, mask):

            # create random positions inside dim range (clustered in the center)
            mu = tf.random.normal([ndim],
                                  mean=0.5,
                                  stddev=0.075)
            mu = tf.multiply(mu, shape)
            # create random sigma values
            sigma = tf.stack([
                sigma_range[0] * shape[i] + tf.random.uniform(
                    []) *
                (sigma_range[1] - sigma_range[0]) * shape[i]
                for i in range(ndim)
            ])

            gauss_function = calc_gauss_function_tf(mu=mu,
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

    #    return tf.multiply(tf.to_float(mask), data), mask    
        #tf.cast(x, tf.float32)
        return tf.multiply(tf.cast(mask, tf.float32), data), mask   







#################################################################################

inputs = [ gt]


LayerMask = CreateMaskLayer()

mask, inputMask = LayerMask(input)

view_slices_3dNew( mask[0,:,:,:,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="mask")

view_slices_3dNew( inputMask[0,:,:,:,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="phase")

##################################################################
####################################################################
