#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:56:30 2024

@author: catarinalopesdias
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:04:02 2024

@author: catarinalopesdias
replicates the add_z_gradient function 
"""
import numpy as np
from numpy import linalg as LA    
from backgroundfieldandeffects.functionsfromsteffen import calc_gauss_function_np, apply_random_brain_mask, distance_to_plane_np, distance_to_plane

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