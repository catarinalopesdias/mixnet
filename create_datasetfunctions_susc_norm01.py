#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:25:45 2023

@author: catarinalopesdias

simulate susceptibility sources
dipole kernel 
forward convolution
"""
import numpy as np
import matplotlib
from matplotlib import transforms
import matplotlib.pyplot as plt


###############################################################################
###############################################################################
###### Simulate susceptibility distribution
###############################################################################
###############################################################################

def simulate_susceptibility_sources_norm(simulation_dim = 128, 
                                    rectangles_total = 80,#800
                                    spheres_total = 80,
                                    sus_std = 1,     # standard deviation of susceptibility values
                                    shape_size_min_factor = 0.01,
                                    shape_size_max_factor = 0.5,
                                    plot=False):
  
    #3d matrix with zeros -size sim dim
  temp_sources = np.zeros((simulation_dim, simulation_dim, simulation_dim))
  
  shrink_factor_all = []
  susceptibility_all = []
  shape_size_all = np.zeros((2,rectangles_total))

  #shapes=0,..., rect total -1
  for shapes in range(rectangles_total):

      # From 1 to almost 0.5       
      shrink_factor = 1/(   (shapes/rectangles_total + 1))
      
      shrink_factor_all.append(shrink_factor)
      
      shape_size_min = np.floor(simulation_dim * shrink_factor * shape_size_min_factor)
      shape_size_max = np.floor(simulation_dim * shrink_factor * shape_size_max_factor)
      

      
      shape_size_all[0,shapes] = shape_size_min
      shape_size_all[1,shapes] = shape_size_max

      ####
      susceptibility_value = np.random.normal(0, 1)
      
      #size of cuboid - random within siye min and max
      random_sizex = np.random.randint(low=shape_size_min, high=shape_size_max)
      random_sizey = np.random.randint(low=shape_size_min, high=shape_size_max)
      random_sizez = np.random.randint(low=shape_size_min, high=shape_size_max)
      
      #position of cuboid (random inside the cube)
      x_pos = np.random.randint(simulation_dim)
      y_pos = np.random.randint(simulation_dim)
      z_pos = np.random.randint(simulation_dim)

      # make sure it does not get out of the cube
      x_pos_max = x_pos + random_sizex
      if x_pos_max >= simulation_dim:
          x_pos_max = simulation_dim

      y_pos_max = y_pos + random_sizey
      if y_pos_max >= simulation_dim:
          y_pos_max = simulation_dim

      z_pos_max = z_pos + random_sizez
      if z_pos_max >= simulation_dim:
          z_pos_max = simulation_dim

      # change the sus values in the cuboids  
      temp_sources[x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max] = susceptibility_value
      susceptibility_all.append(susceptibility_value)
      
      
  #plot
  #if plot:
      #plt.figure(figsize=(12, 4))
  """    plt.figure(figsize=(6, 3))
      fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3)
      plt.suptitle("simulate suscep")
      #plt.subplot(131)
      #plt.plot(np.arange(rectangles_total), shrink_factor_all, ".", color="red")
      ax1.plot(np.arange(rectangles_total), shrink_factor_all, ".", color="red")
      #x1.ylabel("shrink factor")
      #ax1.xlabel("rectangle nr")
      ax1.set(xlabel='rectangle nr', ylabel='shrink factor')
  
      #plt.ylabel("shrink factor")
      #plt.xlabel("rectangle nr")
      #plt.subplot(132)
      ax2.plot(np.arange(rectangles_total), susceptibility_all, ".", color="green")
      #plt.plot(np.arange(rectangles_total), susceptibility_all, ".", color="green")
      #plt.ylabel("susceptibility")
      #plt.xlabel("rectangle nr")
      #ax2.ylabel("susceptibility")
      #ax2.xlabel("rectangle nr")
      ax2.set(xlabel='rectangle nr', ylabel='susceptibility')
  
      #plt.subplot(133)
      ax3.plot(np.arange(rectangles_total),shape_size_all[0,:] , ".", color="green")
      ax3.plot(np.arange(rectangles_total),shape_size_all[1,:] , ".", color="red")
      #plt.plot(np.arange(rectangles_total),shape_size_all[0,:] , ".", color="green")
      #plt.plot(np.arange(rectangles_total),shape_size_all[1,:] , ".", color="red")
      #ax3.ylabel("shape size -min green max red  ")
      #ax3.xlabel("rectangles nr")
      ax3.set(xlabel='rectangle nr', ylabel='shape size -min green max red ')

      #plt.ylabel("shape size -min green max red  ")
      #plt.xlabel("rectangles nr")
"""
  return temp_sources



###############################################################################
###############################################################################
# Convolve Susceptibility Distribution with Dipole Kernel to yield Tissue Phase
###############################################################################
###############################################################################


def generate_3d_dipole_kernel(data_shape, voxel_size, b_vec):
    fov = np.array(data_shape) * np.array(voxel_size)

    ry, rx, rz = np.meshgrid(np.arange(-data_shape[1] // 2, data_shape[1] // 2),
                             np.arange(-data_shape[0] // 2, data_shape[0] // 2),
                             np.arange(-data_shape[2] // 2, data_shape[2] // 2))

    rx, ry, rz = rx / fov[0], ry / fov[1], rz / fov[2]

    sq_dist = rx ** 2 + ry ** 2 + rz ** 2
    sq_dist[sq_dist == 0] = 1e-6
    d2 = ((b_vec[0] * rx + b_vec[1] * ry + b_vec[2] * rz) ** 2) / sq_dist
    kernel = (1 / 3 - d2)

    return kernel


###############################################################################
###############################################################################
def forward_convolution_padding(chi_sample, padding=20):
    #pad sample to avoid wrap-around at the edges
    
    padded_sample = np.zeros((chi_sample.shape[0]+2*padding, chi_sample.shape[1]+2*padding, chi_sample.shape[2]+2*padding))
    padded_sample[padding:chi_sample.shape[0]+padding, padding:chi_sample.shape[1]+padding, padding:chi_sample.shape[2]+padding] = chi_sample
    scaling = np.sqrt(padded_sample.size)
    chi_fft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(padded_sample))) / scaling
    
    dipole_kernel = generate_3d_dipole_kernel(padded_sample.shape, voxel_size=1, b_vec=[0, 0, 1])
    
    chi_fft_t_kernel = chi_fft * dipole_kernel
   
    tissue_phase_unscaled = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(chi_fft_t_kernel)))
    tissue_phase = np.real(tissue_phase_unscaled * scaling)

    tissue_phase_cropped = tissue_phase[padding:chi_sample.shape[0]+padding, padding:chi_sample.shape[1]+padding, padding:chi_sample.shape[2]+padding]
    
    return tissue_phase_cropped
 ###################################################################################### 
def forward_convolution(chi_sample):
    
    scaling = np.sqrt(chi_sample.size)
    chi_fft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(chi_sample))) / scaling
    
    chi_fft_t_kernel = chi_fft * generate_3d_dipole_kernel(chi_sample.shape, voxel_size=1, b_vec=[0, 0, 1])
   
    tissue_phase = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(chi_fft_t_kernel)))
    tissue_phase = np.real(tissue_phase * scaling)

    return tissue_phase
 
###############################################################################

