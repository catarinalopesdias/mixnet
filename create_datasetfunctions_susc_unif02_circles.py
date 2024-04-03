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

def simulate_susceptibility_sources_uni_circles(simulation_dim = 128, 
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
      susceptibility_value = np.random.uniform(low=-0.2, high=0.2)
      
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
  if plot:
      #plt.figure(figsize=(12, 4))
      plt.figure(figsize=(6, 3))
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

  return temp_sources

"""
simulate_susceptibility_sources_uni_circles(simulation_dim = 128, 
                                    rectangles_total = 80,#800
                                    spheres_total = 80,
                                    sus_std = 1,     # standard deviation of susceptibility values
                                    shape_size_min_factor = 0.01,
                                    shape_size_max_factor = 0.5,
                                    plot=True)

simulation_dim =128
sim_gt = np.zeros(( simulation_dim, simulation_dim, simulation_dim))

sim_gt[ :, :, :] = simulate_susceptibility_sources_uni_circles( #simulate_susceptibility_sources_uni
      simulation_dim=simulation_dim, rectangles_total=0, spheres_total=8, plot= False)

"""

from plotting.visualize_volumes import view_slices_3dNew, view_slices_3d

import numpy as np
import random


simulation_dim =128
slicecut = 60


##############
temp_sources3d = np.zeros((simulation_dim, simulation_dim, simulation_dim))

view_slices_3dNew(temp_sources3d, slicecut, slicecut, slicecut, -1,1)

##########
temp_sources2d = np.zeros((simulation_dim, simulation_dim))

plt.imshow(temp_sources2d, vmin=-1, vmax=1, cmap='gray')
##########



nr_circles = 80 
max_radius = 12


# https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle
x = np.arange(0, simulation_dim)
y = np.arange(0, simulation_dim)
#z = np.arange(0.128)
arr = np.zeros((y.size, x.size))

for i in range(nr_circles):
    cxx = random.randint(0, simulation_dim) # “discrete uniform” distribution
    cyy = random.randint(0, simulation_dim)
    rr = random.randint(1, max_radius)



    #####
    # create mask
    maskkk = (x[np.newaxis,:]-cxx)**2 + (y[:,np.newaxis]-cyy)**2 < rr**2
    
    arr[maskkk] = np.random.uniform(low=-0.2, high=0.2)

#plot add
plt.figure(figsize=(6, 6))
plt.pcolormesh(x, y, arr)
plt.set_cmap('gray') #grayscale colormap 
plt.colorbar()
plt.show()

#######################
#######################
#######################
# try 3d 

# https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle
x = np.arange(0, simulation_dim)
y = np.arange(0, simulation_dim)
z = np.arange(0, simulation_dim)



arr3D = np.zeros((x.size, y.size, z.size))



for i in range(1):
    cxx = random.randint(0, simulation_dim)
    cyy = random.randint(0, simulation_dim)
    czz = random.randint(0, simulation_dim)

    rr = random.randint(1, max_radius)

#####

               
    maskkk3D = ((x[ :,         np.newaxis, np.newaxis]  - cxx)**2 + \
               (y[np.newaxis, :         , np.newaxis]  - cyy)**2  + \
               (z[np.newaxis, np.newaxis, :          ] - czz)**2 )< 20**2            
               
    arr3D[maskkk3D] = 20 #np.random.uniform(low=-0.2, high=0.2)

    view_slices_3dNew(arr3D,cxx,cyy,czz, -0.5,0.5)
    #view_slices_3d(arr3D,cxx, -0.5,0.5)

    print("cxx",cxx)
    print("cyy",cyy)
    print("czz",czz)
    print("rr",rr)




#view_slices_3dNew(arr3D,cxx, cyy,czz, -0.5,0.5)

import scipy
def aa(image_3d, slice_nbr_x,slice_nbr_y,slice_nbr_z, vmin, vmax, title=''):

  #image_3d = phantom
  #slice_nbr_x =100
  #slice_nbr_y =100
  #slice_nbr_z = 50
  #vmin=-0.05
  #vmax=0.05
  
  print('input shape', image_3d.shape)

  gridspec = {'width_ratios': [1, 1, 1, 0.1]}
  fig, ax = plt.subplots(1, 4, figsize=(15, 8), gridspec_kw=gridspec) 
  plt.suptitle(title, fontsize=16)

  #print('axial, fixed z shape', np.take(image_3d, slice_nbr_z, 2).shape)

  ax[0].imshow(np.take(image_3d, slice_nbr_z, 2), vmin=vmin, vmax=vmax, cmap='gray')
  ax[0].set_title('Axial: z = ' + str(slice_nbr_z) );
  ax[0].set_xlabel('y')
  ax[0].set_ylabel('x')
  
  ###########################################################
  #coronal
  im_y = np.take(image_3d, slice_nbr_y, 1)
  im_y_rot = im_y #scipy.ndimage.rotate(im_y,90)
  #print('axial, after rotation ', im_y_rot.shape)

  ax[1].imshow(im_y_rot, vmin=vmin, vmax=vmax, cmap='gray')
  ax[1].set_title('Coronal: y= '+str(slice_nbr_y));
  ax[1].set_xlabel('x')
  ax[1].set_ylabel('z')
  ############################################################

  # sagittal 
  im_x = np.take(image_3d, slice_nbr_x, 0)
  print('sagittal, fixed x', im_x.shape)
  im_x_rot = np.swapaxes(im_x,0,1) #scipy.ndimage.rotate(im_x,90)
  print('after rotation ',im_x_rot.shape)


  im = ax[2].imshow(im_x_rot, vmin=vmin, vmax=vmax, cmap='gray')
  ax[2].set_title('Sagittal: x= ' + str(slice_nbr_x));
  ax[2].set_xlabel('y')
  ax[2].set_ylabel('z')

  #######################################################
  # make them square
  ax[1].set_aspect(im_x_rot.shape[1]/im_x_rot.shape[0])
  ax[2].set_aspect(im_x_rot.shape[1]/im_x_rot.shape[0])
  
  cax = ax[3]
  plt.colorbar(im, cax=cax)
  #plt.tight_layout()
  #plt.show()
 
aa(arr3D,cxx,cyy,czz, -0.5,0.5)
    