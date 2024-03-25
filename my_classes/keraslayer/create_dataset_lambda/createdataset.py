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
    
  def __init__(self, output_dim, **kwargs):    
   self.output_dim = output_dim 
   super(MyCustomLayer, self).__init__(**kwargs)

  def build(self, input_shape): 
    self.kernel = self.add_weight(name = 'kernel', 
        shape = (input_shape[1], self.output_dim), 
        initializer = 'normal', trainable = True) 
    super(MyCustomLayer, self).build(input_shape)

    sim_gt[ :, :, :] = simulate_susceptibility_sources(
        simulation_dim=size, rectangles_total=rect_num, plot=False)
    # Phase:forward convolution with the dipole kernel
    sim_fwgt[ :, :, :] = forward_convolution(sim_gt[ :, :, :])


  def call(self, input_data): 
      return K.dot(input_data, self.kernel)

  def compute_output_shape(self, input_shape): return (input_shape[0], self.output_dim)

    