#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:29:22 2023

@author: catarinalopesdias
"""

import numpy as np
import matplotlib
from matplotlib import transforms
import matplotlib.pyplot as plt


# Python 3 Program to check if
# the point lies within the
# ellipse or not
import math
 
# Function to check if point is inside ellipsoid with center cx cy cz
 
 
 
def simulate_susceptibility_sources(simulation_dim = 128, 
                                     rectangles_total = 80,#800
                                     spheres_total = 1,
                                     sus_std = 1,     # standard deviation of susceptibility values
                                     shape_size_min_factor = 0.01,
                                     shape_size_max_factor = 0.5,
                                     plot=False):
   
     #3d matrix with zeros -size sim dim
   temp_sources = np.zeros((simulation_dim, simulation_dim, simulation_dim))
   
   
   ###############################################
   #sphere center
   cx = int(simulation_dim/2)
   cy = int(simulation_dim/2)
   cz = int(simulation_dim/2)
   radius_max = np.floor(simulation_dim/2)
   range=[0.8,0.95] #1 would reach the borders of the cuboid
   radius_x = np.random.random(range(0), range(1) )* radius_max
   radius_y = np.random.random(range(0), range(1) )* radius_max
   radius_z = np.random.random(range(0), range(1) )* radius_max
   
   susceptibility_value_brain = 0.02
   
   if spheres_total ==1:
       temp_sources[cx: cx+radius_x, cy: cy+radius_y, cz: cz+radius_z] = susceptibility_value_brain



#############################
simulation_dim=10
temp_sources = np.zeros((simulation_dim, simulation_dim))

   ###############################################
   #sphere center
cx = int(simulation_dim/2)
cy = int(simulation_dim/2)
radius_max = np.floor(simulation_dim/2)
rangee=[0.8,0.95] #1 would reach the borders of the cuboid
radius_x = int(np.floor(np.random.uniform(rangee[0], rangee[1] )* radius_max))
radius_y = int(np.floor(np.random.uniform(rangee[0], rangee[1] )* radius_max))
   
susceptibility_value_brain = 0.02
bla=cy+radius_y
ble=cx+radius_x


temp_sources[cx: cx+radius_x, cy: cy+radius_y] = susceptibility_value_brain


   ##########################
   #rectangles
   
   
   
   shrink_factor_all = []
   susceptibility_all = []
   shape_size_all = np.zeros((2,rectangles_total))

   #shapes=0,..., rect total -1
   for shapes_rect in range(rectangles_total):

       # From 1 to almost 0.5       
       shrink_factor = 1/(   (shapes_rect/rectangles_total + 1))
       
       shrink_factor_all.append(shrink_factor)
       
       shape_size_min = np.floor(simulation_dim * shrink_factor * shape_size_min_factor)
       shape_size_max = np.floor(simulation_dim * shrink_factor * shape_size_max_factor)
       

       
       shape_size_all[0,shapes_rect] = shape_size_min
       shape_size_all[1,shapes_rect] = shape_size_max

       ####
       susceptibility_value = np.random.normal(loc=0.0, scale=sus_std)
       
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
   
