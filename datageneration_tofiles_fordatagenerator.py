#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:56:39 2024
This follows streffens code for adding background and artifacts
It creates a background based on rectangles (steffen bollman)
and then transforms to phase, and then adds background and artifacts. it saves phase, and phase+bg and artifacts
@author: catarinalopesdias
"""

import h5py
import numpy as np
from plotting.visualize_volumes import view_slices_3dNew
from create_datasetfunctions_susc_unif02 import simulate_susceptibility_sources_uni, forward_convolution
# ,  calc_gauss_function_np#, apply_random_brain_mask, distance_to_plane_np, distance_to_plane
from backgroundfieldandeffects.functionsfromsteffen import apply_random_brain_mask
import backgroundfieldandeffects.functionsfromsteffen 
from backgroundfieldandeffects.generate_backgroundfield_steffen_function import add_z_gradient
from backgroundfieldandeffects.boundaryeffects_function import add_boundary_artifacts

import tensorflow as tf

num_train_instances = 500 
size = 128  # [128,128,128]
rect_num = 200

# Parameters backgroundHHSS
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

testingdata = False
##############################################################################
# create dipole kernel (cuboid size )
##############################################################################
shape_of_sus_cuboid = [size, size, size]  # of the susceptibilitz sources

###############################################################################
# Create synthetic dataset
###############################################################################
# Inizialize 3 steps: gt, phase, background+phase
# 4D-cuboids (training samples, size, size, size)
sim_gt = np.zeros(( size, size, size))
sim_fwgt = np.zeros(( size, size, size))
mask = np.ones_like(sim_gt)
sim_fwgt_mask = np.zeros_like(sim_gt)
bgf = np.zeros_like(mask)
sim_fwgt_mask_bg = np.zeros_like(sim_gt)
sim_fwgt_mask_bg_sn = np.zeros_like(mask)
sim_fwgt_mask_bg_sn_wrapped = np.zeros_like(mask)
bgf_mask = np.zeros_like(mask)
sensornoise = np.zeros_like(mask)

Xinput = np.ones_like(sim_gt)
gtmask = np.ones_like(sim_gt)

Xongoing = np.ones_like(sim_gt)

#################################################################
################################################################
#################################################################
#################################################################

print("iterate epochs,sim susc,  convolve, add background noise")

for epoch_i in range(num_train_instances):
    print("epoch ", str(epoch_i))

    # Create ground thruth - add rectangles
    sim_gt[ :, :, :] = simulate_susceptibility_sources_uni( #simulate_susceptibility_sources_uni
        simulation_dim=size, rectangles_total=rect_num, plot=False)
    # Phase:forward convolution with the dipole kernel
    sim_fwgt[ :, :, :] = forward_convolution(sim_gt[ :, :, :])

 ########################################################################################################
 ########################################################################################################
 ########################################################################################################

    Xinput[ :, :, :] = sim_fwgt[ :, :, :]
    Xongoing[ :, :, :] = sim_fwgt[ :, :, :]

    if apply_masking:

        sim_fwgt_mask[ :, :, :], mask[ :, :,
                                              :] = apply_random_brain_mask(sim_fwgt[ :, :, :])

        Xinput[ :, :, :] = sim_fwgt_mask[ :, :, :]
        
        gtmask[ :, :, :] = tf.multiply(
            sim_gt[ :, :, :], mask[ :, :, :])
        

######################
    if backgroundfield:

        #print("add z gradient")
        bgf[ :, :, :] = add_z_gradient(
            bgf[ :, :, :], gradient_slope_range)

        if apply_masking:

            bgf_mask[ :, :, :] = add_boundary_artifacts(
                bgf[ :, :, :], mask[ :, :, :], boundary_artifacts_mean,
                boundary_artifacts_std)

        # ????????????dkjfsdfsdsdklfjdsjfhhsdgfkjsdfkllfdfjfk
        sim_fwgt_mask_bg[ :, :, :] = tf.add(
            sim_fwgt_mask[ :, :, :], bgf_mask[ :, :, :])

        Xongoing[ :, :, :] = tf.add(
            Xongoing[ :, :, :], bgf_mask[ :, :, :])

######################
    if sensor_noise:

        tf_noise = tf.random.normal(
            shape=[128, 128, 128],  # ]Xongoing[epoch_i,:,:,:].get_shape(),
            mean=sensor_noise_mean,
            stddev=sensor_noise_std,
            dtype=tf.float32)

        sim_fwgt_mask_bg_sn[ :, :, :] = tf.add(
            sim_fwgt_mask_bg[ :, :, :], tf_noise.numpy())

        Xongoing[ :, :, :] = tf.add(
            Xongoing[ :, :, :], tf_noise.numpy())

        sensornoise[ :, :, :] = tf_noise.numpy()
  ######################
    if wrap_input_data:
        # print('wrap_data')
        value_range = 2.0 * np.pi
        # shift from [-pi,pi] to [0,2*pi]
        sim_fwgt_mask_bg_sn_wrapped[ :, :, :] = tf.add(
            sim_fwgt_mask_bg_sn[ :, :, :], value_range / 2.0)

        Xongoing[ :, :, :] = tf.add(
            Xongoing[ :, :, :], value_range / 2.0)

        # # calculate wrap counts
        # self.tensor['wrap_count'] = tf.floor(
        #     tf.divide(X, value_range))

        sim_fwgt_mask_bg_sn_wrapped[ :, :, :] = tf.math.floormod(
            sim_fwgt_mask_bg_sn_wrapped[ :, :, :], value_range)
        Xongoing[ :, :, :] = tf.math.floormod(
            Xongoing[ :, :, :], value_range)

        # shift back to [-pi,pi]
        sim_fwgt_mask_bg_sn_wrapped[ :, :, :] = tf.subtract(
            sim_fwgt_mask_bg_sn_wrapped[ :, :, :], value_range / 2.0)

        Xongoing[ :, :, :] = tf.subtract(
            Xongoing[ :, :, :], value_range / 2.0)

 ######################
    if apply_masking:

        #sim_gt_mask_bg_sn_wrapped = tf.multiply(tf.to_float(mask), sim_gt_mask_bg_sn_wrapped)

        #sim_gt_mask_bg_sn_wrapped = tf.multiply(tf.cast(mask. tf.float32), sim_gt_mask_bg_sn_wrapped)
        sim_fwgt_mask_bg_sn_wrapped[ :, :, :] = np.multiply(
            mask[ :, :, :], sim_fwgt_mask_bg_sn_wrapped[ :, :, :])

        Xongoing[ :, :, :] = np.multiply(
            mask[ :, :, :], Xongoing[ :, :, :])

        #Xongoing = tf.multiply(mask[epoch_i, :,:,:], Xongoing[epoch_i, :,:,:])

        #     Y = tf.multiply(tf.to_float(mask), Y)
    if testingdata:
        file_name = "datasynthetic/uniform02/npz/testing/" + str(epoch_i) + "samples"
    else:
            file_name = "datasynthetic/uniform02/npz/" + str(epoch_i) + "samples"
         
    arr  = np.stack((gtmask,Xinput,Xongoing), axis=0)
            
            #np.save(file_name,arr)
    np.savez_compressed(file_name, arr)


