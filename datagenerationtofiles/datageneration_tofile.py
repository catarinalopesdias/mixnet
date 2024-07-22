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
from create_datasetfunctions import simulate_susceptibility_sources, forward_convolution
# ,  calc_gauss_function_np#, apply_random_brain_mask, distance_to_plane_np, distance_to_plane
from backgroundfieldandeffects.functionsfromsteffen import apply_random_brain_mask
import backgroundfieldandeffects.functionsfromsteffen 
from backgroundfieldandeffects.generate_backgroundfield_steffen_function import add_z_gradient
from backgroundfieldandeffects.boundaryeffects_function import add_boundary_artifacts

import tensorflow as tf

num_train_instances = 115
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
##############################################################################
# create dipole kernel (cuboid size )
##############################################################################
shape_of_sus_cuboid = [size, size, size]  # of the susceptibilitz sources

###############################################################################
# Create synthetic dataset
###############################################################################
# Inizialize 3 steps: gt, phase, background+phase
# 4D-cuboids (training samples, size, size, size)
sim_gt = np.zeros((num_train_instances, size, size, size))
sim_fwgt = np.zeros((num_train_instances, size, size, size))
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
    sim_gt[epoch_i, :, :, :] = simulate_susceptibility_sources(
        simulation_dim=size, rectangles_total=rect_num, plot=False)
    # Phase:forward convolution with the dipole kernel
    sim_fwgt[epoch_i, :, :, :] = forward_convolution(sim_gt[epoch_i, :, :, :])

 ########################################################################################################
 ########################################################################################################
 ########################################################################################################

    Xinput[epoch_i, :, :, :] = sim_fwgt[epoch_i, :, :, :]
    Xongoing[epoch_i, :, :, :] = sim_fwgt[epoch_i, :, :, :]

    if apply_masking:

        sim_fwgt_mask[epoch_i, :, :, :], mask[epoch_i, :, :,
                                              :] = apply_random_brain_mask(sim_fwgt[epoch_i, :, :, :])

        Xinput[epoch_i, :, :, :] = sim_fwgt_mask[epoch_i, :, :, :]
        
        gtmask[epoch_i, :, :, :] = tf.multiply(
            sim_gt[epoch_i, :, :, :], mask[epoch_i, :, :, :])
        

######################
    if backgroundfield:

        #print("add z gradient")
        bgf[epoch_i, :, :, :] = add_z_gradient(
            bgf[epoch_i, :, :, :], gradient_slope_range)

        if apply_masking:

            bgf_mask[epoch_i, :, :, :] = add_boundary_artifacts(
                bgf[epoch_i, :, :, :], mask[epoch_i,
                                            :, :, :], boundary_artifacts_mean,
                boundary_artifacts_std)

        # ????????????dkjfsdfsdsdklfjdsjfhhsdgfkjsdfkllfdfjfk
        sim_fwgt_mask_bg[epoch_i, :, :, :] = tf.add(
            sim_fwgt_mask[epoch_i, :, :, :], bgf_mask[epoch_i, :, :, :])

        Xongoing[epoch_i, :, :, :] = tf.add(
            Xongoing[epoch_i, :, :, :], bgf_mask[epoch_i, :, :, :])

######################
    if sensor_noise:

        tf_noise = tf.random.normal(
            shape=[128, 128, 128],  # ]Xongoing[epoch_i,:,:,:].get_shape(),
            mean=sensor_noise_mean,
            stddev=sensor_noise_std,
            dtype=tf.float32)

        sim_fwgt_mask_bg_sn[epoch_i, :, :, :] = tf.add(
            sim_fwgt_mask_bg[epoch_i, :, :, :], tf_noise.numpy())

        Xongoing[epoch_i, :, :, :] = tf.add(
            Xongoing[epoch_i, :, :, :], tf_noise.numpy())

        sensornoise[epoch_i, :, :, :] = tf_noise.numpy()
  ######################
    if wrap_input_data:
        # print('wrap_data')
        value_range = 2.0 * np.pi
        # shift from [-pi,pi] to [0,2*pi]
        sim_fwgt_mask_bg_sn_wrapped[epoch_i, :, :, :] = tf.add(
            sim_fwgt_mask_bg_sn[epoch_i, :, :, :], value_range / 2.0)

        Xongoing[epoch_i, :, :, :] = tf.add(
            Xongoing[epoch_i, :, :, :], value_range / 2.0)

        # # calculate wrap counts
        # self.tensor['wrap_count'] = tf.floor(
        #     tf.divide(X, value_range))

        sim_fwgt_mask_bg_sn_wrapped[epoch_i, :, :, :] = tf.math.floormod(
            sim_fwgt_mask_bg_sn_wrapped[epoch_i, :, :, :], value_range)
        Xongoing[epoch_i, :, :, :] = tf.math.floormod(
            Xongoing[epoch_i, :, :, :], value_range)

        # shift back to [-pi,pi]
        sim_fwgt_mask_bg_sn_wrapped[epoch_i, :, :, :] = tf.subtract(
            sim_fwgt_mask_bg_sn_wrapped[epoch_i, :, :, :], value_range / 2.0)

        Xongoing[epoch_i, :, :, :] = tf.subtract(
            Xongoing[epoch_i, :, :, :], value_range / 2.0)

 ######################
    if apply_masking:

        #sim_gt_mask_bg_sn_wrapped = tf.multiply(tf.to_float(mask), sim_gt_mask_bg_sn_wrapped)

        #sim_gt_mask_bg_sn_wrapped = tf.multiply(tf.cast(mask. tf.float32), sim_gt_mask_bg_sn_wrapped)
        sim_fwgt_mask_bg_sn_wrapped[epoch_i, :, :, :] = np.multiply(
            mask[epoch_i, :, :, :], sim_fwgt_mask_bg_sn_wrapped[epoch_i, :, :, :])

        Xongoing[epoch_i, :, :, :] = np.multiply(
            mask[epoch_i, :, :, :], Xongoing[epoch_i, :, :, :])

        #Xongoing = tf.multiply(mask[epoch_i, :,:,:], Xongoing[epoch_i, :,:,:])

        #     Y = tf.multiply(tf.to_float(mask), Y)


##############################################################################
# visualize
##############################################################################
index = 1
view_slices_3dNew(gtmask[index, :, :, :], 50, 50, 50,
                  vmin=-1.5, vmax=1.5, title="gt")
view_slices_3dNew(sim_fwgt[index, :, :, :], 50, 50,
                  50, vmin=-1.5, vmax=1.5, title="fw")


view_slices_3dNew(Xinput[index, :, :, :], 50, 50, 50,
                  vmin=-1.5, vmax=1.5, title="Xinput (fw+mask)")
view_slices_3dNew(sim_fwgt_mask[index, :, :, :], 50,
                  50, 50, vmin=-0.5, vmax=0.5, title="fw+brain mask")
view_slices_3dNew(mask[index, :, :, :], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="brain mask")


view_slices_3dNew(bgf[index, :, :, :], 50, 50, 50,
                  vmin=-10, vmax=10, title="background field")
view_slices_3dNew(bgf_mask[index, :, :, :], 50, 50, 50, vmin=-10,
                  vmax=10, title="bg field + masking (bondary artifacts)")
view_slices_3dNew(sim_fwgt_mask_bg[index, :, :, :], 50,
                  50, 50, vmin=-10, vmax=10, title="fw + mask +background")
view_slices_3dNew(sim_fwgt_mask_bg[index, :, :, :], 50, 50, 50, vmin=-
                  0.5, vmax=0.5, title="fw + mask +background+boundarz artifacs")


view_slices_3dNew(sensornoise[index, :, :, :], 50,
                  50, 50, vmin=-0.5, vmax=0.5, title="sn")

view_slices_3dNew(sim_fwgt_mask_bg_sn[index, :, :, :], 50, 50, 50,
                  vmin=-10, vmax=10, title="fw+bg + mask (bondary artifacts)+sn")


view_slices_3dNew(sim_fwgt_mask_bg_sn_wrapped[index, :, :, :], 50, 50, 50,
                  vmin=-10, vmax=10, title="bg field + masking (bondary artifacts)+wrapped")


view_slices_3dNew(Xongoing[index, :, :, :], 50, 50, 50,
                  vmin=-10, vmax=10, title="X ongoing-fw,masj,bg,sn,wrapped")
############################


############################
# gt
#titleGT = "datasynthetic/" + str(num_train_instances)+"GT.npy"

#np.save(titleGT, sim_gt)
##################################
# PHASE BG

#titlephasebg = "datasynthetic/" + str(num_train_instances)+"phase_bg.npy"
#final = "/mnt/neuro/physics/catarina/" + titlephasebg
final1 = "datasynthetic/" + str(num_train_instances)+"phase_bg" + ".h5"

#h5f = h5py.File(final1, 'w')
#h5f.create_dataset('dataset_1', data=Xongoing)
# h5f.close()

#np.save(titlephasebg, Xongoing)
#np.save(final, Xongoing)
#########################################################################
#phase ###############

#titlephase = "datasynthetic/" + str(num_train_instances)+"phase.npy"

#final = "/mnt/neuro/physics/catarina/" + titlephase

#np.save(titlephase, Xinput)
#np.save(titlephase, Xinput)

#final = "/mnt/neuro/physics/catarina/" + titlephase


#/mnt/neuro/nas2/Catarina
title = "datasynthetic/" + str(num_train_instances) + "samples"

np.savez_compressed(title, sim_gt1 = gtmask, phase_bg1 = Xongoing, phase1 = Xinput )
###
#loaded = np.load('')