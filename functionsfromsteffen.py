#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:50:03 2024

@author: catarinalopesdias
"""
import numpy as np

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

def apply_random_brain_mask( data):
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