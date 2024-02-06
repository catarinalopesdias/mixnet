#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:04:02 2024

@author: catarinalopesdias
replicaytes teh add z_gradient 
"""
import os
import tensorflow as tf
import numpy as np
from visualize_volumes import view_slices_3d
from create_datasetfunctions import simulate_susceptibility_sources, generate_3d_dipole_kernel, forward_convolution
from generate_background_field_as_function_christian import generate_backgroundfield
import random
from numpy import linalg as LA    
from functionsfromsteffen import calc_gauss_function_np, apply_random_brain_mask, distance_to_plane_np

num_train = 2
size = 128    #[128,128,128]
rect_num = 30
##############################################################################
# create dipole kernel (cuboid size )
##############################################################################
shape_of_sus_cuboid = [size,size, size] # of the susceptibilitz sources
print("create dipole")
dipole_kernel = generate_3d_dipole_kernel(shape_of_sus_cuboid, voxel_size=1, b_vec=[0, 0, 1])
print("view dipole")
view_slices_3d(dipole_kernel, slice_nbr=50, vmin=-0.5, vmax=0.5, title="dipole kernel")


###############################################################################
# Create synthetic dataset
###############################################################################
# Inizialize 3 steps: gt, phase, background+phase
# 4D-cuboids (training samples, size, size, size)
sim_gt_full =    np.zeros((num_train,size,size,size))
sim_fwgt_full =    np.zeros((num_train,size,size,size)) 
sim_bg_full = np.zeros((num_train,size,size,size)) 
sim_fwbg_full = np.zeros((num_train, size,size,size)) 

# create background field (now it is a cte)
background_fied = generate_backgroundfield() #3D 128 128 128 # this is a constant


##############################################################################
# create synthetic dataset:  
# ground truth: Add rect_num to each cuboid to simulate susceptibility
# Phase: Convolve each cuboid with dipole kernel
#Phase + bg: Add background to each Phase cuboid
##############################################################################
###############################################################################
print("iterate epochs,sim susc,  convolve, add background noise")
for epoch_i in range(num_train):

    # Create ground thruth - add rectangles               
    sim_gt_full[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = rect_num, plot=False)
    # Phase:forward convolution with the dipole kernel 
    sim_fwgt_full[epoch_i,:,:,:]  = forward_convolution(sim_gt_full[epoch_i,:,:,:])
    
    sim_bg_full[epoch_i,:,:,:] = background_fied
    sim_fwbg_full[epoch_i,:,:,:]  = forward_convolution(sim_bg_full[epoch_i, :,:,:])


    # Add background field
    sim_total_full = sim_fwgt_full[epoch_i,:,:,:]  + sim_fwbg_full
###############################################################################
    

###############################################################################
# Visualize synthetic dataset
###############################################################################
#   view ground truth
view_slices_3d(sim_gt_full[1,:,:,:], slice_nbr=50, vmin=-1, vmax=1, title="images of the susceptibility (ground truth)" )
#   view convolution
print("view phase -conv mit dipole")
view_slices_3d(sim_fwgt_full[1,:,:,:], slice_nbr=50, vmin=-1, vmax=1, title= "conv of gt susc sources with dipole kernel")
#   view phase + bavk 

print("view noise")
view_slices_3d(sim_bg_full[1,:,:,:], slice_nbr=50, vmin=-1, vmax=1, title= "Background GT")
print("view noise")
view_slices_3d(sim_fwbg_full[1,:,:,:], slice_nbr=50, vmin=-1, vmax=1, title= "Background Phase")


print("view phase + background field")
view_slices_3d(sim_total_full[1,:,:,:], slice_nbr=50, vmin=-1, vmax=1, title= "total: phase +background phase")
###############################################################################


###############################################################################
###############################################################################
################################################################################
##############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

def distance_to_plane(point, normal, dim, signed_dist=False):
    
    print('distance to plane')
    print("----------------")
    print('point', point)
    print('normal', normal)
    print('dim', dim)
    
    print("----------------")
    
    ndim = len(dim)
    linspace = [np.linspace(0, dim[i] - 1, dim[i]) for i in range(ndim)] 
    
    coord = np.meshgrid(*linspace, indexing='ij')

    print('linspace length', len(linspace))
    print('linspace element 1 size', linspace[1].shape)
    print('coord length', len(coord))
    print('coord [1] shape', coord[1].shape)
    

    coord = np.stack(coord, axis=3)
    
    print('coord after stacking shape', coord.shape)
    
    # calculate distance
    # calculates the dot product between (3,w,h,d) and (3,1)
    pot = point.reshape(3,)
    coord_minus_point = np.subtract(coord, pot)

    
    if signed_dist:
        dist = np.dot(coord_minus_point, normal)
    else:
        dist = np.abs( np.dot(coord_minus_point, normal))
        
    dist = np.squeeze(dist)
        
    dist = np.divide(dist, np.linalg.norm(normal))

    
    return dist

##############################################################################


##############################
##############################################
def add_z_gradient(data, slope_range):
##############################################
    print('add z gradient')
    view_slices_3d(data, slice_nbr=50, vmin=-0.5, vmax=0.5, title="data before bg steffen")
    
    
    #dim = data.shape
    dim = list(data.shape)
    
    #if distance to plane mine
    #point = [i / 2 for i in dim]
    #point = np.zeros((3, 1))
    #for i in range(3):
    #    point[i] = dim[i]/2
    #if distance to plane np    
    point = np.array([i / 2.0 for i in dim]).reshape((3,1))
    
    normal = np.array([0.0, 0.0, -1.0]).reshape((3,1))
    #normal = np.array( [[0.0],[0.0], [-1.0]])
    
    ######################################
    z_jitter = np.random.uniform(low=-dim[2] / 16.0, high=dim[2] / 16.0, size=(1,1) )#size=[1,1]
    
    tmp = np.zeros((2, 1))
    z_jitter =np.append(tmp, z_jitter, axis=0)
    
    
    point = point + z_jitter
    
    print('point with z jitter', point)
    print('xy jitter')
    x_y_jitter = np.random.uniform(
                  low=-0.1, high=0.1, size=(2,1))
    tmp = np.zeros((1, 1))
    
    x_y_jitter = np.append(x_y_jitter, tmp,axis=0)
    
    
    normal = normal + x_y_jitter
    print('normal with z jitter', normal)
    
    normal = normal / LA.norm(normal)
    
    print('normal with z jitter normalized', normal)
    
    
    #dist = 3; 
    dist = distance_to_plane(point, normal, dim, True)
    #dist = distance_to_plane_np(point, normal, dim, True)

    
    print('distance shape', dist.shape)
    #common.distance_to_plane(point, normal, dim, True)
    view_slices_3d(dist, slice_nbr=50, vmin=-100, vmax=100, title="distance")
    
    
    # define slope range
    slope = np.random.uniform(low=slope_range[0] / dim[2],
                              high=slope_range[1] / dim[2])
    dist = dist * slope
    
    view_slices_3d(dist, slice_nbr=50, vmin=-20, vmax=20, title="distance x slope")
    
    # add to data
    data = data + dist
    
    print('data shape', data.shape)
    view_slices_3d(data, slice_nbr=50, vmin=-20, vmax=20, title="data + distance")
    
    return data

####################################################################
slope_range = [3 * 2 * np.pi, 8 * 2 * np.pi]

print('aaaaaaaaaaaaaaaaaaaaaaa')
##############################################
    


##############################################################################
# create synthetic dataset:  
# ground truth: Add rect_num to each cuboid to simulate susceptibility
# Phase: Convolve each cuboid with dipole kernel
#Phase + bg: Add background to each Phase cuboid
##############################################################################
###############################################################################
'''print("iterate epochs,sim susc,  convolve, add background noise")

sim_fancy_full = np.zeros((num_train, size,size,size)) 
sim_fancy_full_fw = np.zeros((num_train, size,size,size)) 

for epoch_i in range(num_train):

    sim_fancy_full[epoch_i,:,:,:] = add_z_gradient( sim_gt_full[epoch_i,:,:,:], slope_range)
    sim_fancy_full_fw[epoch_i,:,:,:]  = forward_convolution(sim_fancy_full[epoch_i,:,:,:])
    

view_slices_3d(sim_gt_full[1,:,:,:], slice_nbr=50, vmin=-0.5, vmax=0.5, title="gt")

view_slices_3d(sim_fancy_full[1,:,:,:], slice_nbr=50, vmin=-0.5, vmax=0.5, title="with z gradient")
    
view_slices_3d(sim_fancy_full_fw[1,:,:,:], slice_nbr=50, vmin=-0.5, vmax=0.5, title="final")
'''    

######################################################
#########################################################



data = sim_gt_full[1,:,:,:]
#####################################################
# data augmentation
##################################################################

     # set default output values
X = sim_gt_full[1,:,:,:]
view_slices_3d(X, slice_nbr=50, vmin=-0.5, vmax=0.5, title="X")

mask = np.ones_like(X)


apply_masking = True
if apply_masking:
    X_input, mask = apply_random_brain_mask(X)
###########################
view_slices_3d(mask, slice_nbr=50, vmin=-0.5, vmax=0.5, title="mask")
view_slices_3d(X_input, slice_nbr=50, vmin=-0.5, vmax=0.5, title="Xinput")


backgroundfield = True
if backgroundfield:

          bgf = np.zeros_like(mask)
          bgf = add_z_gradient(bgf, slope_range)
          
view_slices_3d(bgf, slice_nbr=50, vmin=-0.5, vmax=0.5, title="mask")
          
X = X + bgf          

view_slices_3d(X, slice_nbr=50, vmin=-0.5, vmax=0.5, title="X with bg")
