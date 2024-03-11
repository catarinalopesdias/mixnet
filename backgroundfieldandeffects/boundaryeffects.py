#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:47:22 2024

@author: catarinalopesdias

"""

import numpy
import os
import tensorflow as tf
import numpy as np
from visualize_volumes import view_slices_3d, view_slices_3dNew
from create_datasetfunctions import simulate_susceptibility_sources, generate_3d_dipole_kernel, forward_convolution
from functionsfromsteffen import apply_random_brain_mask, calc_gauss_function_np, calc_gauss_function, forward_simulation
from generate_backgroundfield_steffen import add_z_gradient


def dilation(mask, kernel, kernel_size, shape, num_dilations):
    """Auxiliary function for add_boundary_artifacts().

    Parameters
    ----------
    mask : tf tensor
        Tensor of size HxWxD
    kernel : tf tensor
        Dilation kernel to be used
    kernel_size : list
        List of kernel dimension e.g. [3,3,3]
    shape : tuple
        Size of the data
    num_dilations : int
        Number of dilation steps

    Returns
    -------
    tf tensor
        Result of the dilation

    TODO: move this function somewhere else, maybe common
    """
    X_shape = np.concatenate([[1], shape, [1]])
    #tf.to_float(x) with tf.cast(x, tf.float32)
    #X = tf.reshape(tf.to_float(mask), X_shape)
    maskk =  tf.cast(mask, tf.float32)
    X = tf.reshape(maskk, X_shape)
    
    #self.logger.debug("X: {}".format(X.get_shape()))

    weights_shape = np.concatenate([kernel_size, [1, 1]])
    weights = tf.reshape(kernel, weights_shape)
    #self.logger.debug("weights: {}".format(weights.get_shape()))

    for i in range(num_dilations):

        # convolution
        X = tf.nn.conv3d(input=X,
                         filters=weights,
                         strides=[1, 1, 1, 1, 1],
                         padding='SAME')

        #self.logger.debug("X[{}] {}".format(i, X.get_shape()))

        # thtresholding
        X = tf.cast(X > 0.5, X.dtype)
        #self.logger.debug("X[{}] {} {}".format(i, X.get_shape(), X.dtype))

    dilation_result = tf.reshape(tf.cast(X, tf.bool), shape)

    return dilation_result


def add_boundary_artifacts( data, mask, mean, std):
    """Add artifacts at the mask boundary to the data.

    Parameters
    ----------
    data : tensorflow tensor
        Tensor of size HxWxD
    mask : tensorflow tensor
        Tensor of size HxWxD
    mean : float
    std : float
        Mean and std define the susceptibility range

    Returns
    -------
    tensorflow tensor
        Result of size HxWxD
    """
    shape = list(data.shape)
    #shape = data.get_shape().as_list()

    # number of artifacts
    num_gaussians = 5

    # ellipse size (in terms of gaussian sigma)
    ellipse_size_range = [19.0, 22.0]

    # calculate dilation result
    brain_area = dilation(mask, tf.ones([5, 5, 5]), [5, 5, 5],
                                 shape, 1)

    dilation_result = dilation(brain_area, tf.ones([13, 13, 13]),
                                      [13, 13, 13], shape, 1)

    # selected region for objects
    selected_part = tf.math.logical_xor(dilation_result, brain_area)
#    selected_part = tf.logical_xor(dilation_result, brain_area)

    # only select the lower z region
    zero_z_indices = int(shape[2] * 0.35)
    print("num_of_zeros_z_indices {}".format(zero_z_indices))

    # note that we need to cast here because the slice function is
    # only available for limited datatypes on the GPU!
    selected_part = tf.cast(selected_part, tf.int32)
    selected_part = tf.slice(
        selected_part,
        begin=[0, 0, 0],
        size=[shape[0], shape[1], shape[2] - zero_z_indices])
    selected_part = tf.cast(selected_part, tf.bool)

    print("selected_part shape {}".format(
        selected_part.get_shape()))
    selected_part = tf.concat([
        selected_part,
        tf.zeros([shape[0], shape[1], zero_z_indices], dtype=tf.bool),
    ],
                              axis=2)
    print("selected_part shape {}".format(
        selected_part.get_shape()))

    # get list of indices (which are set 1 in the dilation part)
    list_of_indices = tf.where(tf.equal(selected_part, True))

    print("list_of_indices {}".format(
        list_of_indices.get_shape()))

    # init volume
    artifacts0 = tf.zeros(shape, dtype=tf.float32)
    occupancy_mask0 = tf.zeros(shape, dtype=tf.bool)
    i0 = tf.constant(0)

    def _cond(i, artifacts, occupancy_mask):
        return tf.less(i, num_gaussians)

    def _body(i, artifacts, occupancy_mask):

        # randomly select index from the resulting mask
        
        rnd_idx = tf.random.uniform([],
                                    minval=0.0,
                                    maxval=1.0)
        #tf.random.uniform
        rnd_idx = tf.multiply(rnd_idx,
                              tf.cast(tf.shape(list_of_indices)[0], tf.float32) )
                              #tf.to_float( tf.shape(list_of_indices)[0]))
        #rnd_idx = tf.to_int32(rnd_idx)
        rnd_idx = tf.cast(rnd_idx,tf.int32 )
        mu = list_of_indices[rnd_idx, :]
        #mu = tf.to_float(mu)
        mu = tf.cast(mu,tf.float32)
        print("mu {}".format(mu.get_shape()))

        # add subpixel offsets
        offset = tf.random.uniform(mu.get_shape(),
                                   minval=0.0,
                                   maxval=1.0)
        print("offset {}".format(offset.get_shape()))
        mu = tf.add(mu, offset)

        # randomly select sigma in the given range
        sigma = tf.random.uniform(mu.get_shape(),
                                  minval=ellipse_size_range[0],
                                  maxval=ellipse_size_range[1])

        # randomly select susceptibility in the given range
        susceptibility = tf.random.normal([],
                                          mean=mean,
                                          stddev=std)

        # add gaussians (max value is 1)
        #gauss_function = self.__calc_gauss_function(mu=mu,
        #                                            sigma=sigma,
        #                                            dim=shape)
        
        gauss_function = calc_gauss_function(mu=mu,
                                                    sigma=sigma,
                                                    dim=shape)
        
        

        # threshold to obtain an ellipse
        gauss_function = tf.greater(gauss_function, 0.5)

        # this mask defines the part that doesn't overlap with another ellipse
        current_mask = tf.logical_and(tf.logical_not(occupancy_mask), gauss_function)

        # update occupancy_mask
        occupancy_mask = tf.logical_or(occupancy_mask, current_mask)

        # convert to float
        #gauss_function = tf.to_float(gauss_function)
        gauss_function = tf.cast(gauss_function,tf.float32)

        # apply susceptibility
        gauss_function = tf.multiply(gauss_function, susceptibility)

        # remove overlapping part
        #gauss_function = tf.multiply(tf.to_float(current_mask),      gauss_function)
        gauss_function = tf.multiply(tf.cast(current_mask, tf.float32),      gauss_function)

        # add to result
        artifacts = tf.add(gauss_function, artifacts)

        i += 1
        return (i, artifacts, occupancy_mask)

    i, artifacts, occupancy_mask = tf.while_loop(
        _cond,
        _body,
        loop_vars=(i0, artifacts0, occupancy_mask0),
        parallel_iterations=1,
        name="boundary_artifacts_loop")

    # remove everything that is inside the dilated brain mask


    result = tf.multiply(tf.cast(tf.logical_not(brain_area),tf.float32),
                         artifacts)
    
    # apply forwad simulation
    result, *_ = forward_simulation(result)

    # add artifacts to data
    result = tf.add(data, np.float64(result))

 
    # apply mask again
    result = tf.multiply(tf.cast(mask,tf.float64), result)

    return result

##############################################################################
##############################################################################
# added boundary artifacts (background field with given mask)
boundary_artifacts_mean = 90.0
boundary_artifacts_std = 10.0

# the slope of the added z gradient
z_gradient_range = [3 * 2 * np.pi, 8 * 2 * np.pi]
###############################################################################
num_train = 2
size = 128    #[128,128,128]
rect_num = 30
############################################################################################################################################################
# Create synthetic dataset
sim_gt_full =    np.zeros((num_train,size,size,size))
# ground truth: Add rect_num to each cuboid to simulate susceptibility
##############################################################################
#print("iterate epochs,sim susc,  convolve, add background noise")
#for epoch_i in range(num_train):
    # Create ground thruth - add rectangles               
#    sim_gt_full[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = rect_num, plot=False)

#   view ground truth
#view_slices_3dNew(sim_gt_full[1,:,:,:], 50,50,50, vmin=-1, vmax=1, title="images of the susceptibility (ground truth)" )


###################################################################################

data = sim_gt_full[1,:,:,:]
X = data
##
#view_slices_3d(X, slice_nbr=50, vmin=-0.5, vmax=0.5, title="data X (gt)")
####
X_input, mask = apply_random_brain_mask(X)
####
#view_slices_3d(X_input, slice_nbr=50, vmin=-0.5, vmax=0.5, title="Xinput")
#view_slices_3d(mask, slice_nbr=50, vmin=-0.5, vmax=0.5, title="mask")


####################################
backgroundfield = True
apply_masking = True

if backgroundfield:
                print('backgroundfield')

                bgf = np.zeros_like(mask)
                print('add z gradient')
                    
                bgf = add_z_gradient(bgf,z_gradient_range)
                view_slices_3dNew(bgf, 50,50,50, vmin=-10, vmax=10, title="background field")


                if apply_masking:
                    print("apply masking (and boundary artifacts) to background field")
                            # add boundary artifacts
                            
                    bgf = add_boundary_artifacts(
                                    bgf, mask, boundary_artifacts_mean,
                                    boundary_artifacts_std)
                        
                    view_slices_3dNew(bgf, 50, 50,50, vmin=-10, vmax=10, title="bg field + masking (bondary artifacts)")

                                

                Xin_bg = tf.add(X_input, bgf) #????????????dkjfsdfsdsdklfjdsjfhhsdgfkjsdfkllfdfjfk
                
                view_slices_3d(Xin_bg, slice_nbr=50, vmin=-20, vmax=20, title="bg field + masking")


X= Xin_bg

##########################
wrap_input_data = True



if wrap_input_data:
    print('wrap_data')
    value_range = 2.0 * np.pi
        # shift from [-pi,pi] to [0,2*pi]
    X = tf.add(X, value_range / 2.0)
    
        # # calculate wrap counts
        # self.tensor['wrap_count'] = tf.floor(
        #     tf.divide(X, value_range))
    
    X = tf.math.floormod(X, value_range)
    
        # shift back to [-pi,pi]
    X = tf.subtract(X, value_range / 2.0)
    
    view_slices_3d(X, slice_nbr=50, vmin=-20, vmax=20, title="wrapped")

######################

sensor_noise_mean = 0.0
sensor_noise_std = 0.03

sensor_noise = True
if sensor_noise:
    
    tf_noise = tf.random.normal(
        shape=X.get_shape(),
        mean=sensor_noise_mean,
        stddev=sensor_noise_std,
        dtype=tf.float32)
    
    view_slices_3d(tf_noise.numpy(), slice_nbr=50, vmin=-0.5, vmax=0.5, title="noise")

    X = tf.add(X, tf_noise.numpy()) 
    view_slices_3d(X, slice_nbr=50, vmin=-5, vmax=5, title="noise+X")



###############################################################################
##############################################################################








            
