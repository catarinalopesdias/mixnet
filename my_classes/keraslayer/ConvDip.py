#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convolves 3d image with dipole
"""
from keras import backend as K 
from keras.layers import Layer

import numpy as np
import keras
import tensorflow as tf
# def compute_output_shape(self, input_shape): return (input_shape[0], self.output_dim)

      



class DipConv(Layer):
      

   #########################################################   
  def __init__(self):    
   self.output_dim = input_shape 

   super(MyCustomLayer, self).__init__(**kwargs)

  def build(self, input_shape): 
    self.kernel = self.add_weight(name = 'kernel', 
        shape = (input_shape[1], self.output_dim), 
        initializer = 'zeros', trainable = False) 
    super(MyCustomLayer, self).build(input_shape)

    
  def call(self, input_data): 
      return K.dot(input_data, self.kernel)

#def compute_output_shape(self, input_shape): return (input_shape[0], self.output_dim)
  
    ###############################################################################
###############################################################################
"""def forward_convolution_padding(chi_sample, padding=20):
    #pad sample to avoid wrap-around at the edges
    
    padded_sample = np.zeros((chi_sample.shape[0]+2*padding, chi_sample.shape[1]+2*padding, chi_sample.shape[2]+2*padding))
    padded_sample[padding:chi_sample.shape[0]+padding, padding:chi_sample.shape[1]+padding, padding:chi_sample.shape[2]+padding] = chi_sample
    scaling = np.sqrt(padded_sample.size)
    chi_fft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(padded_sample))) / scaling
    
    dipole_kernel = generate_3d_dipole_kernel(padded_sample.shape, voxel_size=1, b_vec=[0, 0, 1])
    
    chi_fft_t_kernel = chi_fft * dipole_kernel
   
    tissue_phase_unscaled = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(chi_fft_t_kernel)))
    tissue_phase = np.real(tissue_phase_unscaled * scaling)

    tissue_phase_cropped = tissue_phase[padding:chi_sample.shape[0]+padding, padding:chi_sample.shape[1]+padding, padding:chi_sample.shape[2]+padding]
    
    return tissue_phase_cropped"""
 ###################################################################################### 

 
###############################################################################