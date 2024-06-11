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
+  the add_z_gradient_tf 

"""
import numpy as np
from numpy import linalg as LA    
from backgroundfieldandeffects.functionsfromsteffen import   distance_to_plane, distance_to_plane_tf
#from backgroundfieldandeffects.functionsfromsteffen import calc_gauss_function_np, apply_random_brain_mask, distance_to_plane_np, distance_to_plane
import tensorflow as tf
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



    
def add_z_gradient_tf( data, slope_range):
        """Adds a gradient in z direction to the data.

        Parameters
        ----------
        data : tf tensor
            Input data.
        slope_range : list
            List of two elements specifying the min and max range of the
            slope of the z gradient.
        Returns
        -------
        tf tensor
            Input data with the added z gradient.
        """

        dim = data.get_shape().as_list()
        point = tf.constant([i / 2.0 for i in dim],
                            dtype=np.float32,
                            shape=(3, 1))
        normal = tf.constant([0.0, 0.0, -1.0], dtype=np.float32, shape=(3, 1))

        # add jitter for center
        z_jitter = tf.random_uniform(
            (1, 1),
            minval=-dim[2] / 16.0,
            maxval=dim[2] / 16.0)
        tmp = tf.zeros((2, 1), dtype=tf.float32)
        z_jitter = tf.concat([tmp, z_jitter], 0)
        point = tf.add(point, z_jitter)

        # add jitter for normal direction
        x_y_jitter = tf.random_uniform(
            (2, 1),
            minval=-0.1,
            maxval=0.1)
        tmp = tf.zeros((1, 1), dtype=tf.float32)
        x_y_jitter = tf.concat([x_y_jitter, tmp], 0)
        normal = tf.add(normal, x_y_jitter)
        normal = tf.divide(normal, tf.norm(normal))

        # calculate distance to plane
        dist = distance_to_plane_tf(point, normal, dim, True)

        # define slope range
        slope = tf.random_uniform([],
                                  minval=slope_range[0] / dim[2],
                                  maxval=slope_range[1] / dim[2])
        dist = tf.multiply(slope, dist)

        # add to data
        data = tf.add(data, dist)

        return data
    
    
    
def add_z_gradient_SMALL(data, slope_range, reduction=20):
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
        data = data + dist/reduction
        
        #print('data shape', data.shape)
        #view_slices_3d(data, slice_nbr=50, vmin=-20, vmax=20, title="data + (distance*slope)")
        
        return data

    
    