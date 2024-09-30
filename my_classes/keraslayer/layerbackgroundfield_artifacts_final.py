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
from backgroundfieldandeffects.boundaryeffects_function import add_boundary_artifacts


class CreatebackgroundFieldLayer(Layer):
    
    def __init__(self):
        super().__init__()
        self.gradient_slope_range = [3* 2 * tnp.pi, 8 * 2 * tnp.pi] #in tensorflow
        #self.size = 128 
        
    def call(self, inputs):

        print("=== start bgf layer ==============")
        
        bgf_field = False
        boundary_artifacts = True
        wrapping = True

        # make sense of inputs   


        mask =  inputs[0]
        mask =  mask[0,:,:,:,0] # IF 4 D

        #phase
        sim_fwgt_mask = inputs[1]
        sim_fwgt_mask = sim_fwgt_mask[0,:,:,:,0] #IF 4D

        #create background field
        bgf = tf.zeros([128,128,128], tf.float32)


###################################################################################################################################
        if bgf_field:
            bgf_reduction = 5 
            print("add z gradient - reduction = ", bgf_reduction)
            #create bgf        
            bgf = add_z_gradient_SMALL( bgf, self.gradient_slope_range, bgf_reduction) # [ :, :, :] 
        else:
            print("NO z-gradient!")


        ###########################################################################################################
        # bgf with mask
        bgf_mask = tf.multiply(mask, bgf)
        
        #################################################
        #add artifacts
        #########################################
        if boundary_artifacts:
            print("boundary artifacts")

            boundary_artifacts_std = 10.0
            boundary_artifacts_mean = 90.0
            
            bgf_mask = add_boundary_artifacts(
                bgf, mask, boundary_artifacts_mean,
                boundary_artifacts_std)

        else:
            print("no boundary artifacts")
        

        # add background field to the phase 
        sim_fwgt_mask_bg = tf.add(
            sim_fwgt_mask, bgf_mask)


        ##########################
        if wrapping:
            print("wrapping")

            value_range = 2.0 * tnp.pi
            # shift from [-pi,pi] to [0,2*pi]
            
            #add 2pi/2
            sim_fwgt_mask_bg_wrapped = tf.add( sim_fwgt_mask_bg[ :, :, :], value_range / 2.0)

            sim_fwgt_mask_bg_wrapped = tf.math.floormod( sim_fwgt_mask_bg_wrapped, value_range)
            
            # shift back to [-pi,pi]
            sim_fwgt_mask_bg_wrapped = tf.subtract(
                sim_fwgt_mask_bg_wrapped, value_range / 2.0)
            
            sim_fwgt_mask_bg_wrapped = tf.multiply(
                mask, sim_fwgt_mask_bg_wrapped)
            
            output = sim_fwgt_mask_bg_wrapped



            output = sim_fwgt_mask_bg_wrapped
       
        else: 
            print("no wrapping")
            output = sim_fwgt_mask_bg
       #return output_data    

        output = tf.expand_dims(output, 0)
        output = tf.expand_dims(output, 4)
        print("=== end bgf layer ==============")

        return output
    
    
    

#####################################################################################
