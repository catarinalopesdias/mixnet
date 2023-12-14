#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:47:01 2023

@author: catarinalopesdias
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def read_and_decode_tf(example_proto):
    feature_description = {
        'input_data_raw': tf.io.FixedLenFeature([], tf.string),
        'output_data_raw': tf.io.FixedLenFeature([], tf.string),
        'input_height': tf.io.FixedLenFeature([], tf.int64),
        'input_width': tf.io.FixedLenFeature([], tf.int64),
        'input_depth': tf.io.FixedLenFeature([], tf.int64),
        'output_height': tf.io.FixedLenFeature([], tf.int64),
        'output_width': tf.io.FixedLenFeature([], tf.int64),
        'output_depth': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example_proto, feature_description)
    #For dense tensors, the returned Tensor is identical 
    #to the output of parse_example, except there is no batch dimension,
    #the output shape is the same as the shape given in dense_shape.



    input_data = tf.io.decode_raw(example['input_data_raw'], tf.float32)
    output_data = tf.io.decode_raw(example['output_data_raw'], tf.float32)
    #tf.io.decode_raw(input_bytes, out_type, little_endian=True, fixed_length=None, name=None)
    #Every component of the input tensor is interpreted as a sequence of bytes.
    #These bytes are then decoded as numbers in the format specified by out_type.
    input_shape = [
        example['input_height'],
        example['input_width'],
        example['input_depth']
    ]

    output_shape = [
        example['output_height'],
        example['output_width'],
        example['output_depth']
    ]

    input_data = tf.reshape(input_data, input_shape)
    output_data = tf.reshape(output_data, output_shape)

    return input_data, output_data




#########################################
### VISUALIZE RESULTS
#########################################

def visualize_all(resized_input, reference , predicted, title ):
  
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
  ref_min = -1#tf.reduce_min(reference).numpy()
  ref_max = 1#tf.reduce_max(reference).numpy()
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
  
  filename = "images/" + title + ".png"
  plt.savefig(filename)
  plt.show()