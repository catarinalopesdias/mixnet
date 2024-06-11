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
#from create_datasetfunctions_susc_norm01 import simulate_susceptibility_sources_norm, forward_convolution
#create_datasetfunctions import simulate_susceptibility_sources,
# ,  calc_gauss_function_np#, apply_random_brain_mask, distance_to_plane_np, distance_to_plane
from backgroundfieldandeffects.functionsfromsteffen import apply_random_brain_mask
import backgroundfieldandeffects.functionsfromsteffen 
from backgroundfieldandeffects.generate_backgroundfield_steffen_function import add_z_gradient, add_z_gradient_SMALL
from backgroundfieldandeffects.boundaryeffects_function import add_boundary_artifacts

import tensorflow as tf

num_train_instances = 500
size = 128  # [128,128,128]
rect_num = 200

# Parameters backgroundHHSS
gradient_slope_range = [3* 2 * np.pi, 8 * 2 * np.pi]
backgroundfield = True
apply_masking = True

sensor_noise = True

boundary_artifacts_mean = 90.0
boundary_artifacts_std = 10.0


sensor_noise_mean = 0.0
sensor_noise_std = 0.03

sensor_noise = False
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
    
    
    #view_slices_3dNew(sim_gt[ :, :, :], 50, 50, 50,
    #              vmin=-1, vmax=1, title="simgt")

    #view_slices_3dNew(sim_fwgt[ :, :, :], 50, 50, 50,
    #              vmin=-1, vmax=1, title="simfwgt")

 ########################################################################################################
 ########################################################################################################
 ########################################################################################################
#for epoch_i in range(num_train_instances):
#    print("epoch ", str(epoch_i))


    Xinput[ :, :, :] = sim_fwgt[ :, :, :]
    Xongoing[ :, :, :] = sim_fwgt[ :, :, :]

    if apply_masking:

        sim_fwgt_mask[ :, :, :], mask[ :, :,
                                              :] = apply_random_brain_mask(sim_fwgt[ :, :, :])

        Xinput[ :, :, :] = sim_fwgt_mask[ :, :, :]
        
        gtmask[ :, :, :] = tf.multiply(
            sim_gt[ :, :, :], mask[ :, :, :])
        

        #view_slices_3dNew(sim_fwgt_mask[ :, :, :], 50, 50, 50,
        #          vmin=-1, vmax=1, title="sim_fwgt_mask")

        #view_slices_3dNew(gtmask[ :, :, :], 50, 50, 50,
        #          vmin=-1, vmax=1, title="gtmask")

######################
    """if backgroundfield:

    """    
    if testingdata:
        file_name = "datasynthetic/uniform02mask_phase/npz/testing/" + str(epoch_i) + "samples"
    else:
            file_name = "datasynthetic/uniform02mask_phase/npz/" + str(epoch_i) + "samples"
    #     
    arr  = np.stack((mask,sim_fwgt_mask), axis=0)
            
            #np.save(file_name,arr)
    np.savez_compressed(file_name, arr)


#view_slices_3dNew(gtmask[ :, :, :], 50, 50, 50,
#                  vmin=-1.5, vmax=1.5, title="gt+mask")
#view_slices_3dNew(sim_fwgt[index, :, :, :], 50, 50,
 #                 50, vmin=-1.5, vmax=1.5, title="fw")


#view_slices_3dNew(Xinput[index, :, :, :], 50, 50, 50,
#                  vmin=-1.5, vmax=1.5, title="Xinput (fw+mask)")
view_slices_3dNew(sim_fwgt_mask[ :, :, :], 50,
                  50, 50, vmin=-0.2, vmax=0.2, title="fw+brain mask")
#view_slices_3dNew(mask[index, :, :, :], 50, 50, 50,
#                  vmin=-0.5, vmax=0.5, title="brain mask")


#view_slices_3dNew(bgf[ :, :, :], 50, 50, 50,
#                  vmin=-10, vmax=10, title="background field")
view_slices_3dNew(mask[ :, :, :], 50, 50, 50, vmin=-1,
                  vmax=1, title="bg field + mask (bondary artifacts)")
#view_slices_3dNew(sim_fwgt_mask_bg[ :, :, :], 50,
#                  50, 50, vmin=-10, vmax=10, title="fw + mask +background")
#view_slices_3dNew(sim_fwgt_mask_bg[ :, :, :], 50, 50, 50, vmin=-
#                  3.5, vmax=3.5, title="fw + mask +background+boundarz artifacs")


#view_slices_3dNew(sensornoise[ :, :, :], 50,
#                  50, 50, vmin=-0.5, vmax=0.5, title="sn")

#view_slices_3dNew(sim_fwgt_mask_bg_sn[ :, :, :], 50, 50, 50,
#                  vmin=-1.5, vmax=1.5, title="fw+bg + mask (bondary artifacts)+sn")


#view_slices_3dNew(sim_fwgt_mask_bg_sn_wrapped[ :, :, :], 50, 50, 50,
#                  vmin=-3, vmax=3, title="bg field + masking (bondary artifacts)+wrapped")


#view_slices_3dNew(Xongoing[ :, :, :], 50, 50, 50,
#                  vmin=-10, vmax=10, title="X ongoing-fw,masj,bg,sn,wrapped")

#view_slices_3dNew(Xongoing[ :, :, :], 50, 50, 50,
#                  vmin=-3.2, vmax=3.2, title="X ongoing-fw,masj,bg,sn,wrapped")
