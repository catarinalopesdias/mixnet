#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:04:02 2024

@author: catarinalopesdias
replicates the add_z_gradient function 
"""
import numpy as np
from visualize_volumes import view_slices_3d, view_slices_3dNew
from create_datasetfunctions import simulate_susceptibility_sources, forward_convolution
from numpy import linalg as LA    
from functionsfromsteffen import calc_gauss_function_np, apply_random_brain_mask, distance_to_plane_np, distance_to_plane

num_train = 2
size = 128    #[128,128,128]
rect_num = 30
##############################################################################
# create dipole kernel (cuboid size )
##############################################################################
shape_of_sus_cuboid = [size,size, size] # of the susceptibilitz sources
#print("create dipole")
#dipole_kernel = generate_3d_dipole_kernel(shape_of_sus_cuboid, voxel_size=1, b_vec=[0, 0, 1])
#print("view dipole")
#view_slices_3dNew(dipole_kernel, 50,50,50, vmin=-0.5, vmax=0.5, title="dipole kernel")


###############################################################################
# Create synthetic dataset
###############################################################################
# Inizialize 3 steps: gt, phase, background+phase
# 4D-cuboids (training samples, size, size, size)
sim_gt_full =    np.zeros((num_train,size,size,size))
sim_fwgt_full =  np.zeros((num_train,size,size,size)) 


##############################################################################
# create synthetic dataset:  
# ground truth: Add rect_num to each cuboid to simulate susceptibility
# Phase: Convolve each cuboid with dipole kernel
###############################################################################
print("iterate epochs,sim susc,  convolve, add background noise")
for epoch_i in range(num_train):

    # Create ground thruth - add rectangles               
    sim_gt_full[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = rect_num, plot=False)
    # Phase:forward convolution with the dipole kernel 
    sim_fwgt_full[epoch_i,:,:,:]  = forward_convolution(sim_gt_full[epoch_i,:,:,:])
    


###############################################################################
# Visualize synthetic dataset
###############################################################################
#   view ground truth
view_slices_3dNew(sim_gt_full[1,:,:,:], 50,50,50, vmin=-1, vmax=1, title="images of the susceptibility (ground truth)" )
#   view convolution
print("view phase -conv mit dipole")
view_slices_3dNew(sim_fwgt_full[1,:,:,:], 50, 50,50, vmin=-1, vmax=1, title= "conv of gt susc sources with dipole kernel")



##############################
##############################################
def add_z_gradient(data, slope_range):
##############################################
    #print('add z gradient')
    #view_slices_3d(data, slice_nbr=50, vmin=-0.5, vmax=0.5, title="data before bg steffen")
    
    #print('a')
    #dim = data.shape
    dim = list(data.shape)
     
    #point middle
    point = np.array([i / 2.0 for i in dim]).reshape((3,1))
    #normal z dire
    normal = np.array([0.0, 0.0, -1.0]).reshape((3,1))
    
    ######################################
    
    z_jitter = np.random.uniform(low=-dim[2] / 16.0, high=dim[2] / 16.0, size=(1,1) )#size=[1,1]
    tmp = np.zeros((2, 1))
    z_jitter =np.append(tmp, z_jitter, axis=0)
    
    #add z jitter to middle point
    point = point + z_jitter
    #print('point with z jitter', point)
    
    #print('xy jitter')
    #create xy jitter
    x_y_jitter = np.random.uniform(
                  low=-0.1, high=0.1, size=(2,1))
    tmp = np.zeros((1, 1))
    
    x_y_jitter = np.append(x_y_jitter, tmp,axis=0)
    
    #add xy jitter to normal
    normal = normal + x_y_jitter
    #print('normal with z jitter', normal)
    
    #normalize normal 
    normal = normal / LA.norm(normal)
    #print('normal with z jitter normalized', normal)
    
    
    #dist = 3; 
    dist = distance_to_plane(point, normal, dim, True)
    #dist = distance_to_plane_np(point, normal, dim, True)

    
    #print('distance shape', dist.shape)
    #common.distance_to_plane(point, normal, dim, True)
    #view_slices_3d(dist, slice_nbr=50, vmin=-100, vmax=100, title="distance")
    
    
    # define slope range
    slope = np.random.uniform(low=slope_range[0] / dim[2],
                              high=slope_range[1] / dim[2])
    dist = dist * slope
    
    #view_slices_3d(dist, slice_nbr=50, vmin=-20, vmax=20, title="distance x slope")
    
    # add to data
    data = data + dist
    
    #print('data shape', data.shape)
    #view_slices_3d(data, slice_nbr=50, vmin=-20, vmax=20, title="data + (distance*slope)")
    
    return data

####################################################################
####################################################################
####################################################################
# Parameters
slope_range = [3 * 2 * np.pi, 8 * 2 * np.pi]
backgroundfield = True
apply_masking = True
#########################################################

#####################################################
# data augmentation
##################################################################

# set default output values
X = sim_gt_full[1,:,:,:]
#data = X
view_slices_3dNew(X, 50,50,50, vmin=-0.5, vmax=0.5, title="X ground truth")


mask = np.ones_like(X)
if apply_masking:
    X_input, mask = apply_random_brain_mask(X)
    

###########################
view_slices_3dNew(X_input, 50,50,50, vmin=-0.5, vmax=0.5, title="Xinput: data+brain mask")
view_slices_3dNew(mask,50,50,50, vmin=-0.5, vmax=0.5, title="mask")



if backgroundfield:

          bgf = np.zeros_like(mask)
          print("add z gradient")
          bgf = add_z_gradient(bgf, slope_range)
          view_slices_3dNew(bgf, 50,50,50, vmin=-10, vmax=10, title="background")
          
          X = X + bgf  
          view_slices_3dNew(X, 50,50,50, vmin=-10, vmax=10, title="X (gt) with bg")



          
        


