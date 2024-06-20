#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:53:34 2024

@author: catarinalopesdias
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 08:47:22 2024

@author: catarinalopesdias

"""

import numpy
import tensorflow as tf
import numpy as np
from backgroundfieldandeffects.functionsfromsteffen import calc_gauss_function_tf, forward_simulation


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
        
        gauss_function = calc_gauss_function_tf(mu=mu,
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
    result = tf.add(data, tf.cast(result,tf.float32))

 
    # apply mask again
    result = tf.multiply(tf.cast(mask,tf.float32), result)

    return result