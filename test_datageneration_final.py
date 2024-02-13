#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:56:39 2024

@author: catarinalopesdias
"""

import numpy as np
from visualize_volumes import view_slices_3dNew
from create_datasetfunctions import simulate_susceptibility_sources, forward_convolution
from functionsfromsteffen import apply_random_brain_mask #,  calc_gauss_function_np#, apply_random_brain_mask, distance_to_plane_np, distance_to_plane
from generate_backgroundfield_steffen_function import add_z_gradient
from boundaryeffects_function import add_boundary_artifacts

import tensorflow as tf

num_train = 5
size = 128    #[128,128,128]
rect_num = 90

# Parameters background
gradient_slope_range = [3 * 2 * np.pi, 8 * 2 * np.pi]
backgroundfield = True
apply_masking = True

sensor_noise = True

boundary_artifacts_mean = 90.0
boundary_artifacts_std = 10.0


sensor_noise_mean = 0.0
sensor_noise_std = 0.03

sensor_noise = True
wrap_input_data = True
##############################################################################
# create dipole kernel (cuboid size )
##############################################################################
shape_of_sus_cuboid = [size,size, size] # of the susceptibilitz sources

###############################################################################
# Create synthetic dataset
###############################################################################
# Inizialize 3 steps: gt, phase, background+phase
# 4D-cuboids (training samples, size, size, size)
sim_gt =    np.zeros((num_train,size,size,size))
sim_fwgt =  np.zeros((num_train,size,size,size)) 
mask = np.ones_like(sim_gt)
bgf = np.zeros_like(mask)
sim_gt_mask = np.zeros_like(sim_gt)
sim_gt_mask_bg = np.zeros_like(sim_gt)
bgf_mask = np.zeros_like(mask)
sim_gt_mask_bg_sn = np.zeros_like(mask)
sim_gt_mask_bg_sn_wrapped = np.zeros_like(mask)


sensornoise =  np.zeros((num_train,size,size,size)) 

Xinput = np.ones_like(sim_gt)
Xongoing =  np.ones_like(sim_gt)


print("iterate epochs,sim susc,  convolve, add background noise")

for epoch_i in range(num_train):
    print("epoch ", str(epoch_i))
    
    # Create ground thruth - add rectangles               
    sim_gt[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = rect_num, plot=False)
    # Phase:forward convolution with the dipole kernel 
    sim_fwgt[epoch_i,:,:,:]  = forward_convolution(sim_gt[epoch_i,:,:,:])

 ####################################################
 ####################################################
 ####################################################

    Xinput[epoch_i,:,:,:] = sim_gt[epoch_i,:,:,:]
    Xongoing[epoch_i,:,:,:] = sim_gt[epoch_i,:,:,:]
    
    if apply_masking:

        sim_gt_mask[epoch_i,:,:,:], mask[epoch_i,:,:,:] = apply_random_brain_mask(sim_gt[epoch_i,:,:,:])
        
        Xinput[epoch_i,:,:,:] = sim_gt_mask[epoch_i,:,:,:]
        
        
        

    if backgroundfield:

          #print("add z gradient")
          bgf[epoch_i,:,:,:] = add_z_gradient(bgf[epoch_i,:,:,:], gradient_slope_range)
          
          #print("add bg to gt")
          #sim_gt_mask_bg[epoch_i,:,:,:] = sim_gt_mask[epoch_i,:,:,:] + bgf[epoch_i,:,:,:]  
          # Add to gt or to phase, already masked or not?

          if apply_masking:

              bgf_mask[epoch_i,:,:,:] = add_boundary_artifacts(
                  bgf[epoch_i,:,:,:], mask[epoch_i,:,:,:], boundary_artifacts_mean,
                  boundary_artifacts_std)
          
          
          sim_gt_mask_bg[epoch_i,:,:,:] = tf.add(sim_gt_mask[epoch_i,:,:,:], bgf_mask[epoch_i,:,:,:]) #????????????dkjfsdfsdsdklfjdsjfhhsdgfkjsdfkllfdfjfk

          Xongoing[epoch_i,:,:,:] = tf.add(Xongoing[epoch_i,:,:,:], bgf_mask[epoch_i,:,:,:])
          
    if sensor_noise:
        
        tf_noise = tf.random.normal(
            shape=[128,128,128],#]Xongoing[epoch_i,:,:,:].get_shape(),
            mean=sensor_noise_mean,
            stddev=sensor_noise_std,
            dtype=tf.float32)
        
        sim_gt_mask_bg_sn[epoch_i, :,:,:] = tf.add(sim_gt_mask_bg[epoch_i, :,:,:], tf_noise.numpy()) 
        Xongoing[epoch_i,:,:,:] = tf.add(Xongoing[epoch_i, :,:,:], tf_noise.numpy()) 
        
        sensornoise[epoch_i, :,:,:] = tf_noise.numpy()
                
    if wrap_input_data:
         #print('wrap_data')
         value_range = 2.0 * np.pi
                  # shift from [-pi,pi] to [0,2*pi]
         sim_gt_mask_bg_sn_wrapped[epoch_i, :,:,:] = tf.add(sim_gt_mask_bg_sn[epoch_i, :,:,:], value_range / 2.0)
         
         Xongoing[epoch_i, :,:,:] = tf.add(Xongoing[epoch_i, :,:,:], value_range / 2.0)


          # # calculate wrap counts
          # self.tensor['wrap_count'] = tf.floor(
           #     tf.divide(X, value_range))
              
         sim_gt_mask_bg_sn_wrapped[epoch_i, :,:,:] = tf.math.floormod(sim_gt_mask_bg_sn_wrapped[epoch_i, :,:,:], value_range)
         Xongoing[epoch_i, :,:,:] = tf.math.floormod(Xongoing[epoch_i, :,:,:], value_range)
     
         
                  # shift back to [-pi,pi]
         sim_gt_mask_bg_sn_wrapped[epoch_i, :,:,:] = tf.subtract(sim_gt_mask_bg_sn_wrapped[epoch_i, :,:,:], value_range / 2.0)
         
         Xongoing[epoch_i, :,:,:] = tf.subtract(Xongoing[epoch_i, :,:,:], value_range / 2.0)
           
         
    if apply_masking:
              
        #sim_gt_mask_bg_sn_wrapped = tf.multiply(tf.to_float(mask), sim_gt_mask_bg_sn_wrapped)
        
        #sim_gt_mask_bg_sn_wrapped = tf.multiply(tf.cast(mask. tf.float32), sim_gt_mask_bg_sn_wrapped)
        sim_gt_mask_bg_sn_wrapped[epoch_i, :,:,:] = np.multiply(mask[epoch_i, :,:,:], sim_gt_mask_bg_sn_wrapped[epoch_i, :,:,:])

        Xongoing[epoch_i, :,:,:] = np.multiply(mask[epoch_i, :,:,:], Xongoing[epoch_i, :,:,:])

        #Xongoing = tf.multiply(mask[epoch_i, :,:,:], Xongoing[epoch_i, :,:,:])

         #     Y = tf.multiply(tf.to_float(mask), Y)                  
                    




##############################################################################
# visualize
##############################################################################
index = 3
view_slices_3dNew(sim_gt[index,:,:,:], 50,50,50, vmin=-1.5, vmax=1.5, title="gt")
view_slices_3dNew(Xinput[index,:,:,:], 50,50,50, vmin=-1.5, vmax=1.5, title="Xinput")

view_slices_3dNew(sim_fwgt[index,:,:,:], 50,50,50, vmin=-1.5, vmax=1.5, title="phase: gt + conv dipole")
view_slices_3dNew(sim_gt_mask[index,:,:,:], 50,50,50, vmin=-0.5, vmax=0.5, title="Xinput: gt+brain mask")
view_slices_3dNew(mask[index,:,:,:], 50,50,50, vmin=-0.5, vmax=0.5, title="brain mask")
view_slices_3dNew(bgf[index,:,:,:],50,50,50, vmin=-10, vmax=10, title="background field")
view_slices_3dNew(sim_gt_mask_bg[index,:,:,:], 50,50,50, vmin=-10, vmax=10, title="gt + background")
view_slices_3dNew(bgf_mask[index,:,:,:], 50, 50,50, vmin=-10, vmax=10, title="bg field + masking (bondary artifacts)")

view_slices_3dNew(Xongoing[index,:,:,:], 50, 50,50, vmin=-10, vmax=10, title="bg field + masking (bondary artifacts)")





          
          
          
          
 