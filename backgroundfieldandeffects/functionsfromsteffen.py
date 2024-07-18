#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:50:03 2024
Auxiliar functions taken from steffen code
@author: catarinalopesdias
"""
import numpy as np
import tensorflow as tf


def calc_gauss_function_tf( mu, sigma, dim):
    """Calculates a gaussian function (update tensorflow graph).

    Parameters
    ----------
    mu: tuple
        Predefined center of the pdf (x,y,z)
    sigma: float
        Parameter
    dim: tuple
        Dimension of the output (width, height, depth)

    Returns
    -------
    tf tensor
        The desired gaussian function (max value is 1)
    """
    ndim = len(dim)

    linspace = [
        tf.linspace(0.0, dim[i] - 1.0, dim[i]) for i in range(ndim)
    ]
    coord = tf.meshgrid(*linspace, indexing='ij')

    # center at given point
    coord = [coord[i] - mu[i] for i in range(ndim)]

    # create a matrix of coordinates
    coord_col = tf.stack([tf.reshape(coord[i], [-1]) for i in range(ndim)],
                         axis=1)

    covariance_matrix = tf.linalg.diag(tf.square(sigma))

    z = tf.matmul(coord_col, tf.linalg.inv(covariance_matrix))
    z = tf.reduce_sum(tf.multiply(z, coord_col), axis=1)
    z = tf.exp(-0.5 * z)

    # we dont use the scaling
    #z /= np.sqrt(
    #    (2.0 * np.pi)**len(dim) * np.linalg.det(covariance_matrix))

    # reshape to desired size
    z = tf.reshape(z, dim)

    return z




def calc_gauss_function_np( mu, sigma, dim):
        """Calculates a gaussian function (numpy implementation).

        Parameters
        ----------
        mu: tuple
            Predefined center of the pdf (x,y,z)
        sigma: float
            Parameter
        dim: tuple
            Dimension of the output (width, height, depth)

        Returns
        -------
        numpy array
            The desired gaussian function
        """
        linspace = [
            np.linspace(0, dim[i] - 1, dim[i]) for i in range(len(dim))
        ]
        coord = np.meshgrid(*linspace, indexing='ij')

        # center at given point
        coord = [coord[i] - mu[i] for i in range(len(dim))]

        # create a matrix of coordinates
        coord_col = np.column_stack([coord[i].flat for i in range(len(dim))])

        covariance_matrix = np.diag(sigma**2)

        z = np.matmul(coord_col, np.linalg.inv(covariance_matrix))
        z = np.sum(np.multiply(z, coord_col), 1)
        z = np.exp(-0.5 * z)

        # we dont use the scaling
        #z /= np.sqrt(
        #    (2.0 * np.pi)**len(dim) * np.linalg.det(covariance_matrix))

        # reshape to desired sice
        z = z.reshape(dim)

        return z

def apply_random_brain_mask(data):
        """Apply a random brain mask to the data (numpy implementation).

        Parameters
        ----------
        data: 3d numpy array
            Input data

        Returns
        -------
        3D numpy array
            Data with mask applied
        3D numpy array
            mask only
        """

        dim = data.shape

        # number of gaussians to be used
        num_gaussians = 20

        # sigma range in terms of percentage wrt the dim
        sigma_range = [0.15, 0.2]

        # init volume
        mask = np.zeros(dim, dtype=np.float32)

        # create gaussians and sum them up
        for i in range(num_gaussians):

            # create random positions inside dim range (clustered in the center)
            mu = np.random.normal(0.5, 0.075, len(dim)) * dim
            # create random sigma values
            sigma = np.array([
                sigma_range[0] * dim[i] + np.random.rand() *
                (sigma_range[1] - sigma_range[0]) * dim[i]
                for i in range(len(dim))
            ])

            gauss_function = calc_gauss_function_np(mu=mu,
                                                           sigma=sigma,
                                                           dim=dim)

            # update mask
            mask = np.logical_or(gauss_function > 0.5, mask)

        return np.multiply(mask, data), mask
    
    
def apply_random_brain_mask_tf( data):
        """Apply a random brain mask to the data (tensorflow graph update)

        Parameters
        ----------
        data : tf tensor
            Input data of size HxWxD

        Returns
        -------
        tf tensor
            Result of size HxWxD
        tf tensor
            Mask of size HxWxD
        """

        shape = data.get_shape().as_list()
        ndim = len(shape)

        # number of gaussians to be used
        num_gaussians = 20

        # sigma range in terms of percentage wrt the dim
        sigma_range = [0.15, 0.2]

        # init volume
        mask0 = tf.zeros(shape, dtype=tf.bool)

        i0 = tf.constant(0)

        def _cond(i, mask):
            return tf.less(i, num_gaussians)

        def _body(i, mask):

            # create random positions inside dim range (clustered in the center)
            mu = tf.random.normal([ndim],
                                  mean=0.5,
                                  stddev=0.075)
            mu = tf.multiply(mu, shape)
            # create random sigma values
            sigma = tf.stack([
                sigma_range[0] * shape[i] + tf.random.uniform(
                    []) *
                (sigma_range[1] - sigma_range[0]) * shape[i]
                for i in range(ndim)
            ])

            gauss_function = calc_gauss_function_tf(mu=mu,
                                                        sigma=sigma,
                                                        dim=shape)

            # update mask
            mask = tf.logical_or(gauss_function > 0.5, mask)

            i += 1
            return (i, mask)

        i, mask = tf.while_loop(_cond,
                                _body,
                                loop_vars=(i0, mask0),
                                parallel_iterations=1,
                                name="brain_mask_loop")

    #    return tf.multiply(tf.to_float(mask), data), mask    
        #tf.cast(x, tf.float32)
        return tf.multiply(tf.cast(mask, tf.float32), data), mask   
    
def distance_to_plane_np(point, normal, dim, signed_dist=False):
    """Calculates a distance volume to the plane given by a point and a normal.

    Parameters
    ----------
    point: numpy array
        Point on the plane (3d point)
    normal: numpy array
        Normal vector of the plane (3d vector)
    dim: numpy array
        dimension of the volume (x,y,z)
    signed_dist : bool
        Indicates if we should use a signed or unsiged distance

    Returns
    -------
    numpy array
        Array of size dim with distances to the plane
    """
    linspace = [np.linspace(0, dim[i] - 1, dim[i]) for i in range(len(dim))]
    coord = np.meshgrid(*linspace, indexing='ij')
    coord = np.stack(coord, axis=3)

    # calculate distance
    if signed_dist:
        dist = np.dot(np.subtract(coord, point), normal)
    else:
        dist = np.abs(np.dot(np.subtract(coord, point), normal))
    dist = np.divide(dist, np.linalg.norm(normal))

    return dist

def distance_to_plane(point, normal, dim, signed_dist=False):
    
    #print('distance to plane')
    #print("----------------")
    #print('point', point)
    #print('normal', normal)
    #print('dim', dim)
    
    #print("----------------")
    
    ndim = len(dim)
    linspace = [np.linspace(0, dim[i] - 1, dim[i]) for i in range(ndim)] 
    
    coord = np.meshgrid(*linspace, indexing='ij')

    #print('linspace length', len(linspace))
    #print('linspace element 1 size', linspace[1].shape)
    #print('coord length', len(coord))
    #print('coord [1] shape', coord[1].shape)
    

    coord = np.stack(coord, axis=3)
    
    #print('coord after stacking shape', coord.shape)
    
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


def distance_to_plane_tf(point, normal, dim, signed_dist=False):
    """Calculates a distance volume to the plane given by a point and a normal.

    Parameters
    ----------
    point: tf tensor
        Point on the plane (3d point)
    normal: tf tensor
        Normal vector of the plane (3d vector)
    dim: list
        dimension of the volume (x,y,z)
    signed_dist : bool
        Indicates if we should use a signed or unsiged distance

    Returns
    -------
    tf tensor
        Tensor of size dim with distances to the plane
    """
    ndim = len(dim)

    linspace = [tf.linspace(0.0, dim[i] - 1.0, dim[i]) for i in range(ndim)]
    coord = tf.meshgrid(*linspace, indexing='ij')
    coord = [coord[i] - point[i] for i in range(ndim)]
    coord = tf.stack(coord)

    # calculates the dot product between (3,w,h,d) and (3,1)
    dist = tf.tensordot(coord, normal, axes=[[0], [0]])
    dist = tf.reshape(dist, dist.get_shape()[:-1])

    # calculate distance
    if not signed_dist:
        dist = tf.abs(dist)

    dist = tf.divide(dist, tf.linalg.norm(normal))

    return dist

##############################################################################


def calc_dipole_kernel(dim):
        """Calculates the dipole kernel of dimension 'dim'
        (1/3 - k_z^2/k^2)

        Parameters
        ----------
        dim: list
            A list of sampling positions in each dimension

        Returns
        -------
        tf tensor
            Dipole kernel of dimension dim
        """

        linspace = [
            tf.linspace(0.0, float(dim[i] - 1), dim[i])
            for i in range(len(dim))
        ]
        coord = tf.meshgrid(*linspace, indexing='ij')

        # move origin to the center
        coord = [coord[i] - (dim[i] - 1.0) / 2.0 for i in range(len(dim))]

        # calculate squared magnitude
        magnitude = [tf.square(coord[i]) for i in range(len(dim))]
        magnitude = sum(magnitude)
        #magnitude = tf.sqrt(magnitude)

        kernel = 1.0 / 3.0 - tf.square(coord[2]) / magnitude

        return kernel




def fftshift(tensor):
    """Calculates fftshift for a given tf tensor.

    Parameters
    ----------
    tensor : tf-tensor
        Input data

    Retruns
    -------
    tf-tensor
        Tensor after fftshift.
    """
    ndims = len(tensor.get_shape())
    for idx in range(ndims - 1, -1, -1):
        left, right = tf.split(tensor, 2, idx)
        tensor = tf.concat([right, left], idx)
    return tensor

def ifftshift(tensor):
    """Calculates ifftshift for a given tf tensor.

    Parameters
    ----------
    tensor : tf-tensor
        Input data

    Retruns
    -------
    tf-tensor
        Tensor after ifftshift.
    """
    ndims = len(tensor.get_shape())
    for idx in range(ndims):
        left, right = tf.split(tensor, 2, idx)
        tensor = tf.concat([right, left], idx)
    return tensor
    

def forward_simulation( data):
      """Calculates the forward QSM simulation, i.e. apply
      dipole kernel in the fourier domain.

      Parameters
      ----------
      data : tf tensor
          Input data

      Returns
      -------
      tf tensor
          Result
      tf tensor
          Kernel in fourier domain
      tf tensor
          Data in fourier domain
      tf tensor
          Result in fourier domain
      """

      shape = data.get_shape()

      # TODO: padding size should actually be larger
      padding_size = int(shape.as_list()[0] / 8)

      data = tf.pad(
          data, [[padding_size, padding_size], [padding_size, padding_size],
                 [padding_size, padding_size]], "CONSTANT")

      ##############################

      f_kernel = calc_dipole_kernel(data.get_shape().as_list())

      # inverse fft to get the final result
      data_real = data
      data_imag = tf.zeros_like(data)  # tensorflow only supports complex fft
      data = tf.complex(data_real, data_imag)

      f_kernel_real = f_kernel
      f_kernel_imag = tf.zeros_like(f_kernel)
      f_kernel = tf.complex(f_kernel_real, f_kernel_imag)

      f_data = fftshift(tf.signal.fft3d(data))#changed
      f_result = tf.multiply(f_kernel, f_data)
      result = tf.signal.ifft3d(ifftshift(f_result)) #changed

      #############################

      start = tf.constant([padding_size] * 3)
      result = tf.slice(tf.math.real(result), start, shape)
      f_kernel = tf.slice(tf.math.real(f_kernel), start, shape)
      f_data = tf.slice(tf.math.real(f_data), start, shape)
      f_result = tf.slice(tf.math.real(f_result), start, shape)

      return result, f_kernel, f_data, f_result