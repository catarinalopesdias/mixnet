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
from backgroundfieldandeffects.generate_backgroundfield_steffen_function import add_z_gradient_SMALL, add_z_gradient_tf, add_z_gradient
from backgroundfieldandeffects.boundaryeffects_function import add_boundary_artifacts


class CreatebackgroundFieldLayer(Layer):
    
    def __init__(self):
        super().__init__()
        self.gradient_slope_range = [3* 2 * tnp.pi, 8 * 2 * tnp.pi] #in tensorflow
    
        
        
    def call(self, inputs):

        # make sense of inputs   
        
        print("=== start of bgf layer ==========")
        mask =  inputs[0]
        mask =  mask[0,:,:,:,0] # IF 4 D
        
        #phase 
        sim_fwgt_mask = inputs[1]
        sim_fwgt_mask = sim_fwgt_mask[0,:,:,:,0] #IF 4D
        

        ############################################################
        #create background field

        mask_shape = mask.get_shape().as_list()
        bgf = tf.zeros(mask_shape, tf.float32)

        #create bgf        
        bgf = add_z_gradient_SMALL(
            bgf, self.gradient_slope_range, 20) # [ :, :, :] #reduction = 1
        

        #bgf = add_z_gradient_tf( bgf, self.gradient_slope_range)
        #bgf = add_z_gradient( bgf, self.gradient_slope_range, dim)

        ###########################################################################################################
        # bgf with mask
        #bgf_mask = tf.multiply(mask, bgf)
        #add artifacts
        boundary_artifacts_std = 10.0
        boundary_artifacts_mean = 90.0
        
        #boundarz artifacts
        bgf_mask = add_boundary_artifacts(
            bgf, mask, boundary_artifacts_mean,
            boundary_artifacts_std)
        

        #print("bgf mask", bgf_mask)
        #print("phase with mask",sim_fwgt_mask)
        
        # add background field to the phase 
        sim_fwgt_mask_bg = tf.add(
            sim_fwgt_mask, bgf_mask)


        """
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
"""

        #sim_fwgt_mask_bg_wrapped = tf.multiply(
        #    mask, sim_fwgt_mask_bg_wrapped)
        
    
        #output = sim_fwgt_mask_bg_sn_wrapped


        sim_fwgt_mask_bg = tf.expand_dims(sim_fwgt_mask_bg, 0)
        sim_fwgt_mask_bg = tf.expand_dims(sim_fwgt_mask_bg, 4)


        #return sim_fwgt_mask_bg_wrapped
        print("=== end bgf layer =================")

        return sim_fwgt_mask_bg
    
    
    

#####################################################################################
####################################################################################
###################################################################################
######################################################################################
######################################################################################




##################################################################
####################################################################
