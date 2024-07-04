#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 08:56:39 2024
This follows streffens code for adding background and artifacts
It creates a background based on rectangles (steffen bollman)
and then transforms to phase, and then adds background and artifacts. it saves phase, and phase+bg and artifacts
@author: catarinalopesdias
"""


import numpy as np
from plotting.visualize_volumes import view_slices_3dNew
#from create_datasetfunctions_susc_unif02 import simulate_susceptibility_sources_uni, forward_convolution
#from create_datasetfunctions_susc_unif02_circles import simulate_susceptibility_sources_uni, forward_convolution

from create_datasetfunctions_susc_norm01 import forward_convolution
#from create_datasetfunctions_susc_norm01 import simulate_susceptibility_sources_norm, forward_convolution

#create_datasetfunctions import simulate_susceptibility_sources,
# ,  calc_gauss_function_np#, apply_random_brain_mask, distance_to_plane_np, distance_to_plane
from backgroundfieldandeffects.functionsfromsteffen import apply_random_brain_mask
#import backgroundfieldandeffects.functionsfromsteffen 
#from backgroundfieldandeffects.generate_backgroundfield_steffen_function import add_z_gradient, add_z_gradient_SMALL
#from backgroundfieldandeffects.boundaryeffects_function import add_boundary_artifacts

import tensorflow as tf

from create_datasetfunctions_susc_unif02_circles import simulate_susceptibility_sources_1_unirec, simulate_susceptibility_sources_1_unicircle

num_train_instances = 50
size = 128  # [128,128,128]
rect_num = 120
circles_num = 120
# Parameters backgroundHHSS

testingdata = True
#############################################################################
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

gtmask = np.ones_like(sim_gt)


allcircles = np.arange(0, circles_num)
allrects = np.arange(0, rect_num)



#################################################################
################################################################
#################################################################
#################################################################

print("iterate epochs,sim susc,  convolve, add background noise")

for epoch_i in range(num_train_instances):
    print("epoch ", str(epoch_i))



    sim_gt = np.zeros((size, size, size))
    for (circ, rec) in zip(allcircles, allrects):
        simulate_susceptibility_sources_1_unirec(sim_gt)
        simulate_susceptibility_sources_1_unicircle(sim_gt)



    ###############################################
    # Phase:forward convolution with the dipole kernel
    sim_fwgt[ :, :, :] = forward_convolution(sim_gt[ :, :, :])
    
    
    #view_slices_3dNew(sim_gt[ :, :, :], 50, 50, 50,
    #              vmin=-1, vmax=1, title="simgt")

    #view_slices_3dNew(sim_fwgt[ :, :, :], 50, 50, 50,
    #              vmin=-1, vmax=1, title="simfwgt")



    sim_fwgt_mask[ :, :, :], mask[ :, :,
                                              :] = apply_random_brain_mask(sim_fwgt[ :, :, :])


        
    gtmask[ :, :, :] = tf.multiply(
            sim_gt[ :, :, :], mask[ :, :, :])
        

    #view_slices_3dNew(sim_fwgt_mask[ :, :, :], 50, 50, 50,
    #              vmin=-1, vmax=1, title="sim_fwgt_mask")

    #view_slices_3dNew(mask[ :, :, :], 50, 50, 50,
     #             vmin=-1, vmax=1, title="gtmask")

  
    if testingdata:
       file_name = "datasynthetic/uniform02RectCircle_mask_phase/testing/" + str(epoch_i) + "samples"
    else:
       file_name = "datasynthetic/uniform02RectCircle_mask_phase/training/" + str(epoch_i) + "samples"
         
    arr  = np.stack((mask,sim_fwgt_mask), axis=0)
            
    #np.save(file_name,arr)
    np.savez_compressed(file_name, arr)


view_slices_3dNew(mask[ :, :, :], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="gt+mask")
view_slices_3dNew(sim_fwgt_mask[ :, :, :], 50, 50,
                  50, vmin=-0.5, vmax=0.5, title="fw")


#view_slices_3dNew(Xinput[index, :, :, :], 50, 50, 50,
#                  vmin=-1.5, vmax=1.5, title="Xinput (fw+mask)")
#view_slices_3dNew(sim_fwgt_mask[ :, :, :], 50,
         #         50, 50, vmin=-0.2, vmax=0.2, title="fw+brain mask")

