#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:29:06 2023

@author: catarinalopesdias
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import scipy.ndimage

def visualize_all(resized_input, reference , predicted, title, save, path ):
  
  #shape input
  input_data_shape = list(resized_input.shape)
  print("Input shape:", input_data_shape)
  
  #shape reference
  reference_shape = list(reference.shape)
  print("reference shape:", reference_shape)
  
  #shape predicted
  predicted_shape = list(predicted.shape)
  print("predicted shape:", predicted_shape)
  
  
  error = predicted - reference
  
  #error shape
  error_shape = list(error.shape)
  print("error shape:", error_shape)
  
  #cut 3 slices from input 
  #input_slice_x = resized_input[input_data_shape[0]//2, :, :]
  #input_slice_y = resized_input[:, input_data_shape[1]//2, :]
  #input_slice_z = resized_input[:, :, input_data_shape[2]//2]
  
  #cut 3 slices from reference 
  reference_slice_x = reference[reference_shape[0]//2, :, :]
  reference_slice_y = reference[:, reference_shape[1]//2, :]
  reference_slice_z = reference[:, :, reference_shape[2]//2]
  
  #cut 3 slices from predicted
  predicted_slice_x = predicted[predicted_shape[0]//2, :, :]
  predicted_slice_y = predicted[:, predicted_shape[1]//2, :]
  predicted_slice_z = predicted[:, :, predicted_shape[2]//2]
  
  
  # 3 slices of error
  
  error_slice_x = predicted_slice_x - reference_slice_x
  error_slice_y = predicted_slice_y - reference_slice_y
  error_slice_z = predicted_slice_z - reference_slice_z
  
  
  #Get max min of reference
  ref_min =-1.3 #tf.reduce_min(reference).numpy()
  ref_max =1.3 # tf.reduce_max(reference).numpy()
  print("Reference max value", ref_max, "Reference min value", ref_min)

  ####################################################################
  fig = plt.figure(figsize=(10, 10), dpi=100, edgecolor="black" )
  fig.suptitle(title, fontsize=12)

  #plt.title("Model results")
  
  grid = ImageGrid(fig, 311,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )


  grid[0].imshow(reference_slice_x, cmap='gray', vmin = -3.14, vmax = 3.14)
  #grid[0].axis('off')
  grid[0].set_title("Reference data X-dim")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
   
  grid[1].imshow(reference_slice_y, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[1].set_title("Reference data Y-dim")
     
  kk = grid[2].imshow(reference_slice_z, cmap='gray', vmin=ref_min, vmax=ref_max)
  grid[2].set_title("Reference data Z-dim")
   
  grid.cbar_axes[0].colorbar(kk)

   #### predicted
   
  grid = ImageGrid(fig, 312,
      nrows_ncols = (1,3),
      axes_pad = 0.5,
      cbar_location = "right",
      cbar_mode="single",
      cbar_size="5%",
      cbar_pad=1,
      share_all=True
      )

  grid[0].imshow(predicted_slice_x, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[0].set_title("predicted data X-dim ")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(predicted_slice_y, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[1].set_title("predicted data Y-dim ")

  ll = grid[2].imshow(predicted_slice_z, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[2].set_title("predicted data Z-dim")
  grid.cbar_axes[0].colorbar(ll)

  
    
  grid = ImageGrid(fig, 313,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )
   #### error
  
  error_max = 0.3
  error_min = -0.3
  
  
  grid[0].imshow(error_slice_x, cmap='seismic',aspect='equal', vmin=error_min, vmax=error_max)
  grid[0].set_title("Error data X-dim")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(error_slice_y, cmap='seismic',aspect='equal', vmin=error_min, vmax=error_max)
  grid[1].set_title("Error data Y-dim")


  jj = grid[2].imshow(error_slice_z, cmap='seismic',aspect='equal', vmin=error_min, vmax=error_max)
  grid[2].set_title("Error data Z-dim ")
  grid.cbar_axes[0].colorbar(jj)
  if save:      
      filename = path +  ".png"
      plt.savefig(filename)
  plt.show()
       
  return [predicted_slice_x, predicted_slice_y, predicted_slice_z],  [reference_slice_x, reference_slice_y, reference_slice_z], [error_slice_x, error_slice_y, error_slice_z]






########################################################################
# plots 3 views of a 3d image
##########################################################

def view_slices_3d(image_3d, slice_nbr, vmin, vmax, title=''):
#   print('Matrix size: {}'.format(image_3d.shape))
  fig = plt.figure(figsize=(8, 5))
  plt.suptitle(title, fontsize=8)

  plt.subplot(131)
  plt.imshow(np.take(image_3d, slice_nbr, 2), vmin=vmin, vmax=vmax, cmap='gray')
  plt.title('Axial-fix z');

  plt.subplot(132)
  #image_rot = ndimage.rotate(np.take(image_3d, slice_nbr, 1),90)
  image_rot = np.take(image_3d, slice_nbr, 1)
  plt.imshow(image_rot, vmin=vmin, vmax=vmax, cmap='gray')
  plt.title('Coronal fix y');

  plt.subplot(133)
  #image_rot = ndimage.rotate(np.take(image_3d, slice_nbr, 0),90)
  image_rot = np.take(image_3d, slice_nbr, 0)

  plt.imshow(image_rot, vmin=vmin, vmax=vmax, cmap='gray')
  plt.title('Sagittal - fix x');
  cbar=plt.colorbar()
  
  ##########################################################
  
  
def view_slices_3dNew(image_3d, slice_nbr_x,slice_nbr_y,slice_nbr_z, vmin, vmax, title=''):

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

  print('axial, fixed z shape', np.take(image_3d, slice_nbr_z, 2).shape)

  ax[0].imshow(np.take(image_3d, slice_nbr_z, 2), vmin=vmin, vmax=vmax, cmap='gray')
  ax[0].set_title('Axial: z = ' + str(slice_nbr_z) );
  ax[0].set_xlabel('y')
  ax[0].set_ylabel('x')
  
  ###########################################################
  im_y = np.take(image_3d, slice_nbr_y, 1)
  print('axial fixed y ', im_y.shape)
  im_y_rot = np.swapaxes(im_y,0,1) # scipy.ndimage.rotate(im_y,90)
  print('axial, after rotation ', im_y_rot.shape)

  ax[1].imshow(im_y_rot, vmin=vmin, vmax=vmax, cmap='gray')
  ax[1].set_title('Coronal: y= '+str(slice_nbr_y));
  ax[1].set_xlabel('x')
  ax[1].set_ylabel('z')
  ############################################################

  im_x = np.take(image_3d, slice_nbr_x, 0)
  print('sagittal, fixed x', im_x.shape)
  im_x_rot = np.swapaxes(im_x,0,1)  #scipy.ndimage.rotate(im_x,90)
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
 
#  if save:    
 #      filename = path +  ".png"
  #      plt.savefig(filename)
  #plt.show()
    
  #return [predicted_slice_x, predicted_slice_y, predicted_slice_z],  [reference_slice_x, reference_slice_y, reference_slice_z], [error_slice_x, error_slice_y, error_slice_z]


 
def visualize_all4(resized_input, reference , predicted, title, save, path, colormax=1, colormin=-1, errormax=0.2, errormin=-0.2):
  
  #shape input
  input_data_shape = list(resized_input.shape)
  print("Input shape:", input_data_shape)
  
  #shape reference
  reference_shape = list(reference.shape)
  print("reference shape:", reference_shape)
  
  #shape predicted
  predicted_shape = list(predicted.shape)
  print("predicted shape:", predicted_shape)
  
  
  error = predicted - reference
  
  #error shape
  error_shape = list(error.shape)
  print("error shape:", error_shape)
  
  #cut 3 slices from input 
  input_slice_x = resized_input[input_data_shape[0]//2, :, :]
  input_slice_y = resized_input[:, input_data_shape[1]//2, :]
  input_slice_z = resized_input[:, :, input_data_shape[2]//2]
  
  #cut 3 slices from reference 
  reference_slice_x = reference[reference_shape[0]//2, :, :]
  reference_slice_y = reference[:, reference_shape[1]//2, :]
  reference_slice_z = reference[:, :, reference_shape[2]//2]
  
  #cut 3 slices from predicted
  predicted_slice_x = predicted[predicted_shape[0]//2, :, :]
  predicted_slice_y = predicted[:, predicted_shape[1]//2, :]
  predicted_slice_z = predicted[:, :, predicted_shape[2]//2]
  
  
  # 3 slices of error
  
  error_slice_x = predicted_slice_x - reference_slice_x
  error_slice_y = predicted_slice_y - reference_slice_y
  error_slice_z = predicted_slice_z - reference_slice_z
  
  
  #Get max min of reference
  ref_min =-1#1.3 #tf.reduce_min(reference).numpy()
  ref_max =1#1.3 # tf.reduce_max(reference).numpy()
  print("Reference max value", ref_max, "Reference min value", ref_min)

  ####################################################################
  fig = plt.figure(figsize=(10, 10), dpi=100, edgecolor="black" )
  fig.suptitle(title, fontsize=12)

  ###########################################################
  grid = ImageGrid(fig, 411,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )
   #### error
  

  
  
  grid[0].imshow(input_slice_x, cmap='gray',aspect='equal', vmin=-15, vmax=15)
  grid[0].set_title("Background Field - X")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(input_slice_y, cmap='gray',aspect='equal', vmin=-15, vmax=15)
  grid[1].set_title("Background Field - Y")


  jj = grid[2].imshow(input_slice_z, cmap='gray',aspect='equal', vmin=-15, vmax=15)
  grid[2].set_title("Input data - Z")
  grid.cbar_axes[0].colorbar(jj)

  ########################################################################
  grid = ImageGrid(fig, 412,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )


  grid[0].imshow(reference_slice_x, cmap='gray', vmin = colormin, vmax = colormax)
  #grid[0].axis('off')
  grid[0].set_title("Reference - X")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
   
  grid[1].imshow(reference_slice_y, cmap='gray',  vmin=colormin, vmax=colormax)
  grid[1].set_title("Reference - Y")
     
  kk = grid[2].imshow(reference_slice_z, cmap='gray', vmin=colormin, vmax=colormax)
  grid[2].set_title("Reference - Z")
   
  grid.cbar_axes[0].colorbar(kk)

  #### predicted
  #################################################################### 
  grid = ImageGrid(fig, 413,
      nrows_ncols = (1,3),
      axes_pad = 0.5,
      cbar_location = "right",
      cbar_mode="single",
      cbar_size="5%",
      cbar_pad=1,
      share_all=True
      )

  grid[0].imshow(predicted_slice_x, cmap='gray',  vmin=colormin, vmax=colormax)
  grid[0].set_title("Predicted - X")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(predicted_slice_y, cmap='gray',  vmin=colormin, vmax=colormax)
  grid[1].set_title("Predicted - Y")

  ll = grid[2].imshow(predicted_slice_z, cmap='gray',  vmin=colormin, vmax=colormax)
  grid[2].set_title("Predicted - Z")
  grid.cbar_axes[0].colorbar(ll)

  
###############################################################################################
  grid = ImageGrid(fig, 414,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )
   #### error
  
  #error_max = 0.3
  #error_min = -0.3
  
  
  grid[0].imshow(error_slice_x, cmap='seismic',aspect='equal', vmin=errormin, vmax=errormax)
  grid[0].set_title("Difference - X")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(error_slice_y, cmap='seismic',aspect='equal', vmin=errormin, vmax=errormax)
  grid[1].set_title("Difference - Y")


  jj = grid[2].imshow(error_slice_z, cmap='seismic',aspect='equal', vmin=errormin, vmax=errormax,)
  grid[2].set_title("Difference - Z")
  grid.cbar_axes[0].colorbar(jj)

  if save:    
      filename = path +  ".pdf"
      plt.savefig(filename)
  plt.show()
  
  return [predicted_slice_x, predicted_slice_y, predicted_slice_z],  [reference_slice_x, reference_slice_y, reference_slice_z], [error_slice_x, error_slice_y, error_slice_z]



############################################################################

def visualize_all4grey(resized_input, reference , predicted, title, save, path, colormax, colormin, errormax, errormin, slice_nr=64 ):
  
  #shape input
  input_data_shape = list(resized_input.shape)
  print("Input shape:", input_data_shape)
  
  #shape reference
  reference_shape = list(reference.shape)
  print("reference shape:", reference_shape)
  
  #shape predicted
  predicted_shape = list(predicted.shape)
  print("predicted shape:", predicted_shape)
  
  
  error = predicted - reference
  
  #error shape
  error_shape = list(error.shape)
  print("error shape:", error_shape)
  

  #cut 3 slices from input 
  input_slice_x = resized_input[slice_nr, :, :]
  input_slice_y = resized_input[:, slice_nr, :]
  input_slice_z = resized_input[:, :, slice_nr]
  
  #cut 3 slices from reference 
  reference_slice_x = reference[slice_nr, :, :]
  reference_slice_y = reference[:, slice_nr, :]
  reference_slice_z = reference[:, :, slice_nr]
  
  #cut 3 slices from predicted
  predicted_slice_x = predicted[slice_nr, :, :]
  predicted_slice_y = predicted[:, slice_nr, :]
  predicted_slice_z = predicted[:, :, slice_nr]
  
  
  # 3 slices of error
  
  error_slice_x = predicted_slice_x - reference_slice_x
  error_slice_y = predicted_slice_y - reference_slice_y
  error_slice_z = predicted_slice_z - reference_slice_z
  
  
  #Get max min of reference
  ref_min =colormin#1.3 #tf.reduce_min(reference).numpy()
  ref_max =colormax#1.3 # tf.reduce_max(reference).numpy()
  print("Reference max value", colormax, "Reference min value", colormin)

  ####################################################################
  fig = plt.figure(figsize=(10, 10), dpi=100, edgecolor="black" )
  fig.suptitle(title, fontsize=12)

  ###########################################################
  grid = ImageGrid(fig, 411,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )
   #### error
  
 
  
  
  grid[0].imshow(input_slice_x, cmap='gray',aspect='equal', vmin=ref_min, vmax=ref_max)
  grid[0].set_title("Input data X-dim")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(input_slice_y, cmap='gray',aspect='equal', vmin=ref_min, vmax=ref_max)
  grid[1].set_title("Input data Y-dim")


  jj = grid[2].imshow(input_slice_z, cmap='gray',aspect='equal', vmin=ref_min, vmax=ref_max)
  grid[2].set_title("Input data Z-dim ")
  grid.cbar_axes[0].colorbar(jj)

  ########################################################################
  grid = ImageGrid(fig, 412,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )


  grid[0].imshow(reference_slice_x, cmap='gray', vmin = ref_min, vmax = ref_max)
  #grid[0].axis('off')
  grid[0].set_title("Reference data X-dim")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
   
  grid[1].imshow(reference_slice_y, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[1].set_title("Reference data Y-dim")
     
  kk = grid[2].imshow(reference_slice_z, cmap='gray', vmin=ref_min, vmax=ref_max)
  grid[2].set_title("Reference data Z-dim")
   
  grid.cbar_axes[0].colorbar(kk)

  #### predicted
  #################################################################### 
  grid = ImageGrid(fig, 413,
      nrows_ncols = (1,3),
      axes_pad = 0.5,
      cbar_location = "right",
      cbar_mode="single",
      cbar_size="5%",
      cbar_pad=1,
      share_all=True
      )

  grid[0].imshow(predicted_slice_x, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[0].set_title("predicted data X-dim ")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(predicted_slice_y, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[1].set_title("predicted data Y-dim ")

  ll = grid[2].imshow(predicted_slice_z, cmap='gray',  vmin=ref_min, vmax=ref_max)
  grid[2].set_title("predicted data Z-dim")
  grid.cbar_axes[0].colorbar(ll)

  
###############################################################################################
  grid = ImageGrid(fig, 414,
     nrows_ncols = (1,3),
     axes_pad = 0.5,
     cbar_location = "right",
     cbar_mode="single",
     cbar_size="5%",
     cbar_pad=1,
     share_all=True
     )
   #### error
  

  
  grid[0].imshow(error_slice_x, cmap='gray',aspect='equal', vmin=errormin, vmax=errormax)
  grid[0].set_title("Error data X-dim")
  grid[0].get_xaxis().set_ticks([])
  grid[0].get_yaxis().set_ticks([])
    
  grid[1].imshow(error_slice_y, cmap='gray',aspect='equal', vmin=errormin, vmax=errormax)
  grid[1].set_title("Error data Y-dim")


  jj = grid[2].imshow(error_slice_z, cmap='gray',aspect='equal', vmin=errormin, vmax=errormax)
  grid[2].set_title("Error data Z-dim ")
  grid.cbar_axes[0].colorbar(jj)

  if save:    
      filename = path +  ".png"
      plt.savefig(filename)
  plt.show()
  
  return [predicted_slice_x, predicted_slice_y, predicted_slice_z],  [reference_slice_x, reference_slice_y, reference_slice_z], [error_slice_x, error_slice_y, error_slice_z]
