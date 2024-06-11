#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:55:37 2024

@author: catarinalopesdias
"""

from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np

from tensorflow.experimental import numpy as tnp # start using tnp instead of numpy or math library#test tnp.pi, tnp.e
from backgroundfieldandeffects.generate_backgroundfield_steffen_function import add_z_gradient_SMALL
    

class CreatePhaseLayer(Layer):
    
    def __init__(self):
        super().__init__()
        

        
    #def build(self, input_shape):   
        
        
    def call(self, inputs):
        #num_rows, num_columns = tf.shape(inputs)[1], tf.shape(inputs)[2]

        # make sense of inputs   
        
        
        mask =  inputs[0]
        mask =  mask[0,:,:,:,0] # IF 4 D

               sim_fwgt[ :, :, :] = forward_convolution(sim_gt[ :, :, :])
        def forward_convolution(chi_sample):
    
    scaling = np.sqrt(chi_sample.size)
    chi_fft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(chi_sample))) / scaling
    
    chi_fft_t_kernel = chi_fft * generate_3d_dipole_kernel(chi_sample.shape, voxel_size=1, b_vec=[0, 0, 1])
   
    tissue_phase = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(chi_fft_t_kernel)))
    tissue_phase = np.real(tissue_phase * scaling)

    return tissue_phase





        sim_fwgt_mask_bg_wrapped = tf.expand_dims(sim_fwgt_mask_bg_wrapped, 0)
        sim_fwgt_mask_bg_wrapped = tf.expand_dims(sim_fwgt_mask_bg_wrapped, 4)
       #return output_data    
        return sim_fwgt_mask_bg_wrapped
    
    
    

#####################################################################################
####################################################################################
###################################################################################
######################################################################################
######################################################################################


num_train_instances = 500
size = 128  # [128,128,128]
rect_num = 200

# Parameters backgroundHHSS
gradient_slope_range = [3* 2 * np.pi, 8 * 2 * np.pi]
backgroundfield = True
apply_masking = True

sensor_noise = True

boundary_artifacts_mean = 90.0
boundary_artifacts_std = 10.0


sensor_noise_mean = 0.0
sensor_noise_std = 0.03

sensor_noise = False
wrap_input_data = True

testingdata = False
##############################################################################
# create dipole kernel (cuboid size )
##############################################################################
shape_of_sus_cuboid = [size, size, size]  # of the susceptibilitz sources

###############################################################################
# Create synthetic dataset
###############################################################################
# Inizialize 3 steps: gt, phase, background+phase
# 4D-cuboids (training samples, size, size, size)
sim_gt = np.zeros(( size, size, size))
sim_fwgt = np.zeros(( size, size, size))
mask = np.ones_like(sim_gt)
sim_fwgt_mask = np.zeros_like(sim_gt)
bgf = np.zeros_like(mask)
sim_fwgt_mask_bg = np.zeros_like(sim_gt)
sim_fwgt_mask_bg_sn = np.zeros_like(mask)
sim_fwgt_mask_bg_sn_wrapped = np.zeros_like(mask)
bgf_mask = np.zeros_like(mask)
sensornoise = np.zeros_like(mask)

Xinput = np.ones_like(sim_gt)
gtmask = np.ones_like(sim_gt)

Xongoing = np.ones_like(sim_gt)

#################################################################
################################################################
#################################################################
#################################################################

print("iterate epochs,sim susc,  convolve, add background noise")

for epoch_i in range(num_train_instances):
    print("epoch ", str(epoch_i))

    # Create ground thruth - add rectangles
    sim_gt[ :, :, :] = simulate_susceptibility_sources_uni( #simulate_susceptibility_sources_uni
        simulation_dim=size, rectangles_total=rect_num, plot=False)
    # Phase:forward convolution with the dipole kernel
    sim_fwgt[ :, :, :] = forward_convolution(sim_gt[ :, :, :])







    ############################################################################
    ######################################################################
    ####################################################################################
    ################################################################################













from plotting.visualize_volumes import view_slices_3dNew


from create_datasetfunctions_susc_unif02 import simulate_susceptibility_sources_uni, forward_convolution
from backgroundfieldandeffects.functionsfromsteffen import apply_random_brain_mask
size = 128  
rect_num = 200


#gt
sim_gt = simulate_susceptibility_sources_uni( #simulate_susceptibility_sources_uni
    simulation_dim=size, rectangles_total=rect_num, plot=False)

#phase
sim_fwgt = forward_convolution(sim_gt[ :, :, :])

### apply masks
# phase with mask
sim_fwgt_mask, mask = apply_random_brain_mask(sim_fwgt[ :, :, :])
#gt with mask
gtmask = tf.multiply(sim_gt[ :, :, :], mask[ :, :, :])


#
# Should I expand the dimension??

gtmask = np.expand_dims(gtmask, 0)
sim_fwgt_mask = np.expand_dims(sim_fwgt_mask, 0) 
mask = np.expand_dims(mask, 0)


gtmask = np.expand_dims(gtmask, 4)
sim_fwgt_mask = np.expand_dims(sim_fwgt_mask, 4) 
mask = np.expand_dims(mask, 4)



#convert to tensorflow
sim_fwgt_mask = tf.convert_to_tensor(sim_fwgt_mask)
   # value, dtype=Non
mask = tf.convert_to_tensor(mask,dtype=tf.float32 )
mask = tf.cast(mask, tf.float32)


#### view  add 0 in last dimenstion if dim 4 
view_slices_3dNew(gtmask[ 0,:, :, :,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="gtmask")

view_slices_3dNew(sim_fwgt_mask[ 0,:, :, :,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="phase with mask")

view_slices_3dNew(mask[0, :, :, :,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title=" mask")


## input for layer - phase with mask, mask




inputs = [ mask, sim_fwgt_mask]


LayerWBakgroundField = CreatebackgroundFieldLayer()

output = LayerWBakgroundField(inputs)

view_slices_3dNew( output[0,:,:,:,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="phase with background field and mask")


##################################################################
####################################################################


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


def generate_3d_dipole_kernel_tf(data_shape, voxel_size, b_vec):
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