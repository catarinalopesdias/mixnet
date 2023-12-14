#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# new preprocessing + training
"""
Created on Wed Dec  6 09:28:17 2023

@author: catarinalopesdias
"""

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import matplotlib
from matplotlib import transforms
from scipy import ndimage
import os
import tensorflow as tf


from keras.layers import Input ,Conv3D, Conv3DTranspose, LeakyReLU, UpSampling3D, Concatenate , Add, BatchNormalization
from keras.models import Model
from keras.initializers import GlorotNormal
from keras.callbacks import ModelCheckpoint
import pickle

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
  




###############################################################################
###############################################################################
###### Simulate susceptibility distribution
###############################################################################
###############################################################################

def simulate_susceptibility_sources(simulation_dim = 160, 
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
  if plot:
      plt.figure(figsize=(12, 4))
      plt.title("simulate suscep")
      plt.subplot(131)
      plt.plot(np.arange(rectangles_total), shrink_factor_all, ".", color="red")
      plt.ylabel("shrink factor")
      plt.xlabel("rectangle nr")
      plt.subplot(132)
      plt.plot(np.arange(rectangles_total), susceptibility_all, ".", color="green")
      plt.ylabel("susceptibility")
      plt.xlabel("rectangle nr")
  
      plt.subplot(133)
      plt.plot(np.arange(rectangles_total),shape_size_all[0,:] , ".", color="green")
      plt.plot(np.arange(rectangles_total),shape_size_all[1,:] , ".", color="red")
      plt.ylabel("shape size -min green max red  ")
      plt.xlabel("rectangles nr")

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

num_epochs = 50
size = 128#[128,128,128]

# create dipole kernel

shape_of_sus_cuboid = [size,size, size] # of the susceptibilitz sources
print("create dipole")
dipole_kernel = generate_3d_dipole_kernel(shape_of_sus_cuboid, voxel_size=1, b_vec=[0, 0, 1])
print("view dipole")
view_slices_3d(dipole_kernel, slice_nbr=100, vmin=-0.5, vmax=0.5, title="dipole kernel")


sim_gt_full = np.zeros((num_epochs,size,size,size))
sim_fw_full = np.zeros((num_epochs,size,size,size)) 

print("iterate epochs,sim susc,  convolve")
for epoch_i in range(num_epochs):
    if epoch_i ==1:
        plotH=True
    else:
         plotH=False
                   
    sim_gt_full[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = 80, plot=plotH)
    # forward convolution with the dipole kernel 
    sim_fw_full[epoch_i,:,:,:]  = forward_convolution(sim_gt_full[epoch_i,:,:,:])



view_slices_3d(sim_gt_full[2,:,:,:], slice_nbr=100, vmin=-1, vmax=1, title="images of the susceptibility (ground truth)" )

print("view phase -conv mit dipole")
view_slices_3d(sim_fw_full[2,:,:,:], slice_nbr=100, vmin=-1, vmax=1, title= "conv of gt susc sources with dipole kernel")


###############################################################################
##############################################################################


def build_CNN(input_tensor):


#########################################################################################################################
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(input_tensor)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc3 = Add()([X_save, X])

    #down convolutional layer
    encoding_down_1 = Conv3D(filters=16,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2))(X_conc3)
    batch_norm_layer_3 = BatchNormalization()(encoding_down_1)
    # batch_norm_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_3)                    #i dont know if i need that-check it

#############################################################################################################################
    #Second Layer-->down
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc2 = Add()([X_save, X])


    encoding_down_2 = Conv3D(filters=32,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2)
                            )(X_conc2)
    batch_norm_layer_2 = BatchNormalization()(encoding_down_2)
    # batch_norm_layer_2 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_2)
###################################################################################################################################################
    #Third Layer-->down
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_2)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc1 = Add()([X_save, X])
    encoding_down_3 = Conv3D(filters=64,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2)
                            )(X_conc1)

    batch_norm_layer_3 = BatchNormalization()(encoding_down_3)
    # batch_norm_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_3)
#######################################################################################################################
    #Fourth Layer = Connection between Layer
    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

#####################################################################################################
    #Third Layer --> up NOte: if not workin, you have to slice
    decoder_up_1 = UpSampling3D(size=(2,2,2))(X)
    decoder_1 = Conv3DTranspose(filters=64,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_1)

    #vertical connection between these convolutional layers--> adds the output together
    #TensorShape([None, 3, 32, 32, 64])


    combine_conc1_dec1 = Concatenate()([X_conc1, decoder_1])

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(combine_conc1_dec1)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
##########################################################################################################
    #Second layer -->up
    decoder_up_2 = UpSampling3D(size=(2,2,2))(X)
    decoder_2 = Conv3DTranspose(filters=32,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_2)

    #vertical connection between these convolutional layers--> adds the output together

    combine_conc2_dec2 = Concatenate()([X_conc2, decoder_2])

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(combine_conc2_dec2)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
########################################################################################################################
    # Third Layer --> up
    decoder_up_3 = UpSampling3D(size=(2,2,2))(X)
    decoder_3 = Conv3DTranspose(filters=16,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_3)

    combine_conc3_dec3 = Concatenate()([X_conc3, decoder_3])

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(combine_conc3_dec3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
###################################################################################################################
    # output before residual conenction
    output_layer = Conv3D(filters=1 ,kernel_size=[3,3,3], padding='same')(X)

    #residual connection between input and output
    residual_conn = Add()([input_tensor, output_layer])
    output_tensor=residual_conn

    #r#eturn output_tensor
    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

#############################################################
#### Compile the model
#############################################################
input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")

#ushape1 = build_CNN(input_tensor)
#ushape2 = build_CNN(ushape1)

print("compile model")

#model = Model(input_tensor, ushape2) #get from orig deepQSM algorithm
model = build_CNN(input_tensor)
model.compile(optimizer='adam', loss='mean_absolute_error')
model.summary()
###############################################################################
###############################################################################
print("untrained")
# what does the untrained model predict
test_epoch_nbr = 10
X_test = sim_fw_full[np.newaxis, test_epoch_nbr,:,:,:, np.newaxis]
print(X_test.shape)

y_pred = model.predict(X_test)

print(y_pred.shape)

view_slices_3d(X_test[0, :, :, :, 0], slice_nbr=16, vmin=-1, vmax=1, title='Input Tissue Phase')
view_slices_3d(sim_gt_full[test_epoch_nbr, :, :, :], slice_nbr=16, vmin=-1, vmax=1, title='GT Susceptibility')
view_slices_3d(y_pred[0, :, :, :, 0], slice_nbr=16, vmin=-1, vmax=1, title='Predicted Susceptibility')

################################################################################
# training part
##########################################################################



epochs_train = 50
save_period = 20
batch_size=1
###############

# train
checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=save_period,
                                                 verbose=1)

train_images =tf.expand_dims(sim_fw_full, 4)
train_labels = tf.expand_dims(sim_gt_full, 4)

print("fit model")
history = model.fit(train_images, train_labels,  epochs=epochs_train, batch_size=batch_size, shuffle=True,
          callbacks = [cp_callback])  # pass callback to training for saving the model

loss_history = history.history['loss']


with open('loss_history.pickle', 'wb') as f:
    pickle.dump([loss_history, epochs_train], f)
    
    
#save model
if not os.path.exists("models"): 
    os.makedirs("models") 
    
model_name = "models/model_" + str(num_epochs) +"epochs_" + "batchsize"+ str(batch_size)+".h5"
#model.save(model_name)
tf.keras.models.save_model(model, model_name)    
    
    
##################################################################################################
##################################################################################################

from  visualizenetworkprediction import visualize_all
## create test set

num_epochs = 5
size = 128#[128,128,128]

#

test_sim_gt_full = np.zeros((num_epochs,size,size,size)) 

test_sim_fw_full = np.zeros((num_epochs,size,size,size)) 

print("iterate epochs,sim susc,  convolve")
for epoch_i in range(num_epochs):
    if epoch_i ==1:
        plotH=True
    else:
         plotH=False
                   
    test_sim_gt_full[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = 80, plot=False)
    # forward convolution with the dipole kernel 
    test_sim_fw_full[epoch_i,:,:,:]  = forward_convolution(test_sim_gt_full[epoch_i,:,:,:])
#############################



print("trained")
# what does the untrained model predict

for epoch_i in range(num_epochs):
    X_test = test_sim_fw_full[np.newaxis, epoch_i,:,:,:, np.newaxis]

    y_pred = model.predict(X_test)

    print(epoch_i)
        
    visualize_all(test_sim_fw_full[epoch_i,:,:,:], test_sim_gt_full[epoch_i,:,:,:], y_pred[0,:,:,:,0] , "network" )

    #view_slices_3d(X_test[0, :, :, :, 0], slice_nbr=16, vmin=-1, vmax=1, title='Input Tissue Phase (test)')
    #view_slices_3d(test_sim_gt_full[test_epoch_nbr, :, :, :], slice_nbr=16, vmin=-1, vmax=1, title='GT Susceptibility (test)')
    #view_slices_3d(y_pred[0, :, :, :, 0], slice_nbr=16, vmin=-1, vmax=1, title='Predicted Susceptibility aftyer training')





