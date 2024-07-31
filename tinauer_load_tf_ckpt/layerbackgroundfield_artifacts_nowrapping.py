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
        #self.size = 128 
        
    #def build(self, input_shape):   
        
        
    def call(self, inputs):

        # make sense of inputs   
        
        #num_rows, num_columns = tf.shape(inputs)[1], tf.shape(inputs)[2]

        #print("ssss", tf.shape(inputs[1]))

        mask =  inputs[0]
        mask =  mask[0,:,:,:,0] # IF 4 D
        
        #phase 
        sim_fwgt_mask = inputs[1]
        sim_fwgt_mask = sim_fwgt_mask[0,:,:,:,0] #IF 4D
        
        #print("shape MASK",mask.get_shape().as_list())
 #       print("shape MASK",tf.shape(tf.gather(mask)))

        print("shape MASK orig",tf.shape(mask))      
        dim_mask = tf.shape(mask)
            #print("shape MASK",tf.shape(mask))
        #tf.print("shape MASK",mask.get_shape().as_list())
        #tf.print("shape MASK",tf.shape(mask))

        #print(len(mask.numpy()))

        #print(tf.cast(mask,tf.float32))
        #print("dfds", tf.shape(inputs)[0], tf.shape(inputs)[1])
        #phase 
        #sim_fwgt_mask = inputs[1]
        #sim_fwgt_mask = sim_fwgt_mask[0,:,:,:,0] #IF 4D
        #print("shape phase orig",sim_fwgt_mask.shape.as_list())   
        ############################################################
        ############################################################
       # print("here")
        #bla = tf.print(tf.shape(mask[0,:,0])[0])
        #bla = eval(tf.shape(mask[0,:,0])[0]))
        #print("bla")
        #tf.print(bla)
      
        dim_mask1= tf.shape(mask[0,:,0])[0]
        #print("tf print dim_mask")
        #tf.print(dim_mask1)
        #print("tf print dim_mask",tf.print(dim_mask1) )
        #bgf = tf.zeros([dim_mask1,dim_mask1,dim_mask1], tf.float32)
        bgf = tf.zeros([128,128,128], tf.float32)

        
        #bgf = tf.zeros_like(mask)
        #tf.print(bgf)     
        
        
        #bgf = tf.zeros_like(mask, tf.float32)
        #bgf = tf.zeros_like(mask)
        #tf.zeros([self.size , self.size ,self.size ], tf.float32)
        

        #print("shape tensor bgf - self",tf.shape(bgf))

        #bgf = tf.zeros_like(mask)
        
        #tf.print("bgf before input",bgf.shape.as_list())   

###################################################################################################################################
        #create background field
        #bgf = tf.zeros_like(mask)
        #tf.zeros([self.size , self.size ,self.size ], tf.float32)

        #create bgf        
        #bgf = add_z_gradient_SMALL(
         #   bgf, self.gradient_slope_range, 1) # [ :, :, :] #reduction = 1
        

        bgf = add_z_gradient_tf( bgf, self.gradient_slope_range)
        #bgf = add_z_gradient( bgf, self.gradient_slope_range, dim)

        ###########################################################################################################
        # bgf with mask
        #bgf_mask = tf.multiply(mask, bgf)
        #add artifacts
        boundary_artifacts_std = 10.0
        boundary_artifacts_mean = 90.0
        bgf_mask = add_boundary_artifacts(
            bgf, mask, boundary_artifacts_mean,
            boundary_artifacts_std)
        

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


       #return output_data    
        #return sim_fwgt_mask_bg_wrapped
        return sim_fwgt_mask_bg
    
    
    

#####################################################################################
####################################################################################
###################################################################################
######################################################################################
######################################################################################




##################################################################
####################################################################
