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
import random
from plotting.visualize_volumes import view_slices_3dNew, view_slices_3d


###############################################################################
###############################################################################
###### Simulate susceptibility distribution
###############################################################################
###############################################################################

def simulate_susceptibility_sources_uni_rec(simulation_dim = 128, 
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

      ax1.plot(np.arange(rectangles_total), shrink_factor_all, ".", color="red")
      ax1.set(xlabel='rectangle nr', ylabel='shrink factor')
      ax2.plot(np.arange(rectangles_total), susceptibility_all, ".", color="green")
      ax2.set(xlabel='rectangle nr', ylabel='susceptibility')
      #plt.subplot(133)
      ax3.plot(np.arange(rectangles_total),shape_size_all[0,:] , ".", color="green")
      ax3.plot(np.arange(rectangles_total),shape_size_all[1,:] , ".", color="red")

      ax3.set(xlabel='rectangle nr', ylabel='shape size -min green max red ')

  return temp_sources




#######################################################################################
#######################################################################################
#######################################################################################



#bla = simulate_susceptibility_sources_uni_rec(simulation_dim = 128, 
#                                    rectangles_total = 80,#800
#                                    spheres_total = 80,
#                                    sus_std = 1,     # standard deviation of susceptibility values
#                                    shape_size_min_factor = 0.01,
#                                    shape_size_max_factor = 0.5,
#                                    plot=True)
  
#view_slices_3dNew(bla,60, 60,60, -0.5,0.5)



#######################
#######################
#######################


###########################################
###########################################
###########################################


# try 3d 
# https://stackoverflow.com/questions/49330080/numpy-2d-array-selecting-indices-in-a-circle
def simulate_susceptibility_sources_1_unicircle(volume):
    simulation_dim = volume.shape[0]
    max_radius = simulation_dim / 8
    x = np.arange(0, simulation_dim)
    y = np.arange(0, simulation_dim)
    z = np.arange(0, simulation_dim)
    
    cxx = random.randint(10, simulation_dim-10) #10
    cyy = random.randint(10, simulation_dim-10) #50
    czz = random.randint(10, simulation_dim-10) #100
    rr = random.randint(3, max_radius)

    maskkk3D = ((x[ :,         np.newaxis, np.newaxis]  - cxx)**2 + \
               (y[np.newaxis, :         , np.newaxis]  - cyy)**2  + \
               (z[np.newaxis, np.newaxis, :          ] - czz)**2 )< rr**2            
               
    susceptibility = np.random.uniform(low=-0.2, high=0.2)    
    volume[maskkk3D] = volume[maskkk3D] + susceptibility
    
    return volume




###################################################################
#dim = 128
#arr3D = np.zeros((dim, dim, dim))
#nr_circles = 80
#for i in range(nr_circles):
#    simulate_susceptibility_sources_1_unicircle(arr3D)

#view_slices_3dNew(arr3D,dim/2, dim/2,dim/2, -0.5,0.5)


###############################################
###############################################
###############################################

def simulate_susceptibility_sources_1_unirec(volume):
  simulation_dim = volume.shape[0]

  shape_size_min = 2#np.floor(simulation_dim * shrink_factor * shape_size_min_factor)
  shape_size_max = 50#np.floor(simulation_dim * shrink_factor * shape_size_max_factor)


      ####
  susceptibility_value = np.random.uniform(low=-0.2, high=0.2)
      
      #size of cuboid - random within siye min and max
  [sizex,sizey,sizez] = np.random.randint(shape_size_min, shape_size_max,3)
  random_sizey = np.random.randint(shape_size_min, shape_size_max)
  random_sizez = np.random.randint(shape_size_min, shape_size_max)
      
      #position of cuboid (random inside the cube)
  [x_pos,y_pos, z_pos] = np.random.randint(10, simulation_dim-10,3)
  #y_pos = np.random.randint(10, simulation_dim-10)
  #z_pos = np.random.randint(10, simulation_dim-10)

      # make sure it does not get out of the cube
  x_pos_max = min(x_pos + sizex, simulation_dim-10)
  #np.random.randint(x_pos, simulation_dim-10)
  y_pos_max = min(y_pos + sizey, simulation_dim-10)
  #np.random.randint(y_pos, simulation_dim-10)#min(y_pos + sizey, simulation_dim)
  z_pos_max = min(z_pos + sizez, simulation_dim-10)
  #np.random.randint(z_pos, simulation_dim-10)#min(z_pos + sizez, simulation_dim)

    
  #for i in range(3):
  # check= False
  # while check:
     #  ar = np.sort(np.random.randint(10, simulation_dim-10,2))
       
   #    diff = ar[1]- ar[0]
    #   if diff < 50:
     #      check=True
   
  #[x_min, x_max] = ar #np.sort(np.random.randint(10, simulation_dim-10,2))
  
  #check= False
  #while check:
   #    ar = np.sort(np.random.randint(10, simulation_dim-10,2))
       
    #   diff = ar[1]- ar[0]
     #  if diff < 50:
      #     check=True
           
  #[y_min, y_max] = ar#np.sort(np.random.randint(10, simulation_dim-10,2))
  
  #check= False
  #while check:
   #    ar = np.sort(np.random.randint(10, simulation_dim-10,2))
       
    #   diff = ar[1]- ar[0]
     #  if diff < 50:
      #     check=True
  
  #[y_min, z_max] = ar#np.sort(np.random.randint(10, simulation_dim-10,2))


      # change the sus values in the cuboids
  volume[x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max] = volume[x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max] + susceptibility_value

  return volume

###########################################
#dim = 128
#nr_rect = 80

#arr3D = np.zeros((dim, dim, dim))

#for i in range(nr_rect):
#    simulate_susceptibility_sources_1_unirec(arr3D)

#view_slices_3dNew(arr3D,dim/2, dim/2,dim/2, -0.5,0.5)
#################################################
###################
#nr_circles = 80
#nr_rect = 80

#############
#allcircles = np.arange(0, nr_circles)
#allrects = np.arange(0, nr_rect)
#dim= 128
#arr3D = np.zeros((dim, dim, dim))
###############################################################################

#for (circ, rec) in zip(allcircles, allrects):
#    simulate_susceptibility_sources_1_unirec(arr3D)
#    simulate_susceptibility_sources_1_unicircle(arr3D)
    
    
#view_slices_3dNew(arr3D,dim/2, dim/2,dim/2, -0.5,0.5)

#######################################################################################
#########################'
#######################################################################################
#########################'
##############################################################################