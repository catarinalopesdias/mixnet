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
    

class CreatebackgroundFieldLayer(Layer):
    
    def __init__(self):
        super().__init__()
        self.gradient_slope_range = [3* 2 * tnp.pi, 8 * 2 * tnp.pi] #in tensorflow

        
    #def build(self, input_shape):   
        
        
    def call(self, inputs):
        #num_rows, num_columns = tf.shape(inputs)[1], tf.shape(inputs)[2]

        # make sense of inputs   
        
        
        mask =  inputs[0]
        mask =  mask[0,:,:,:,0] # IF 4 D

              
        
        sim_fwgt_mask = inputs[1]
        sim_fwgt_mask = sim_fwgt_mask[0,:,:,:,0] #IF 4D

        #create background field
        bgf = tf.zeros([size, size,size], tf.float32)

        #create bgf        
        bgf = add_z_gradient_SMALL(
            bgf, self.gradient_slope_range) # [ :, :, :]


        # bgf with mask
        bgf_mask = tf.multiply(mask, bgf)
        

        # add background field to the phase 
        sim_fwgt_mask_bg = tf.add(
            sim_fwgt_mask, bgf_mask)



        value_range = 2.0 * tnp.pi
        # shift from [-pi,pi] to [0,2*pi]
        
        #sim_fwgt_mask_bg_sn = sim_fwgt_mask_bg

        #add 2pi/2
        sim_fwgt_mask_bg_wrapped = tf.add( sim_fwgt_mask_bg[ :, :, :], value_range / 2.0)

        sim_fwgt_mask_bg_wrapped = tf.math.floormod( sim_fwgt_mask_bg_wrapped, value_range)
        
        # shift back to [-pi,pi]
        sim_fwgt_mask_bg_wrapped = tf.subtract(
            sim_fwgt_mask_bg_wrapped, value_range / 2.0)
        
        sim_fwgt_mask_bg_wrapped = tf.multiply(
            mask, sim_fwgt_mask_bg_wrapped)
        
    
        #output = sim_fwgt_mask_bg_sn_wrapped


        sim_fwgt_mask_bg_wrapped = tf.expand_dims(sim_fwgt_mask_bg_wrapped, 0)
        sim_fwgt_mask_bg_wrapped = tf.expand_dims(sim_fwgt_mask_bg_wrapped, 4)
       #return output_data    
        return sim_fwgt_mask_bg_wrapped
    
    
    

#####################################################################################
####################################################################################
###################################################################################
######################################################################################
######################################################################################


from plotting.visualize_volumes import view_slices_3dNew


from create_datasetfunctions_susc_unif02 import simulate_susceptibility_sources_uni, forward_convolution
from backgroundfieldandeffects.functionsfromsteffen import apply_random_brain_mask
size = 128  
rect_num = 200


#gt
sim_gt = simulate_susceptibility_sources_uni( #simulate_susceptibility_sources_uni
    simulation_dim=size, rectangles_total=rect_num, plot=False)

#phase
sim_fwgt = forward_convolution(sim_gt[ :, :, :])

### apply masks
# phase with mask
sim_fwgt_mask, mask = apply_random_brain_mask(sim_fwgt[ :, :, :])
#gt with mask
gtmask = tf.multiply(sim_gt[ :, :, :], mask[ :, :, :])


#
# Should I expand the dimension??

gtmask = np.expand_dims(gtmask, 0)
sim_fwgt_mask = np.expand_dims(sim_fwgt_mask, 0) 
mask = np.expand_dims(mask, 0)


gtmask = np.expand_dims(gtmask, 4)
sim_fwgt_mask = np.expand_dims(sim_fwgt_mask, 4) 
mask = np.expand_dims(mask, 4)



#convert to tensorflow
sim_fwgt_mask = tf.convert_to_tensor(sim_fwgt_mask)
   # value, dtype=Non
mask = tf.convert_to_tensor(mask,dtype=tf.float32 )
mask = tf.cast(mask, tf.float32)


#### view  add 0 in last dimenstion if dim 4 
view_slices_3dNew(gtmask[ 0,:, :, :,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="gtmask")

view_slices_3dNew(sim_fwgt_mask[ 0,:, :, :,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="phase with mask")

view_slices_3dNew(mask[0, :, :, :,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title=" mask")


## input for layer - phase with mask, mask




inputs = [ mask, sim_fwgt_mask]


LayerWBakgroundField = CreatebackgroundFieldLayer()

output = LayerWBakgroundField(inputs)

view_slices_3dNew( output[0,:,:,:,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="phase with background field and mask")


##################################################################
####################################################################
