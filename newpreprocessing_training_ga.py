#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# new preprocessing + training +gradient accumulation
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

from newGA import GradientAccumulateModel
from network import build_CNN
from create_datasetfunctions import simulate_susceptibility_sources, generate_3d_dipole_kernel, forward_convolution
from visualizenetworkprediction import view_slices_3d

#############################################################


num_train = 100
size = 128#[128,128,128]
rect_num = 30

# create dipole kernel

shape_of_sus_cuboid = [size,size, size] # of the susceptibilitz sources
print("create dipole")
dipole_kernel = generate_3d_dipole_kernel(shape_of_sus_cuboid, voxel_size=1, b_vec=[0, 0, 1])
print("view dipole")
view_slices_3d(dipole_kernel, slice_nbr=50, vmin=-0.5, vmax=0.5, title="dipole kernel")


sim_gt_full = np.zeros((num_train,size,size,size))
sim_fw_full = np.zeros((num_train,size,size,size)) 

print("iterate epochs,sim susc,  convolve")
for epoch_i in range(num_train):
    #if epoch_i == 1:
    plotH=False
    #else:
     #    plotH=False
                   
    sim_gt_full[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = rect_num, plot=plotH)
    # forward convolution with the dipole kernel 
    sim_fw_full[epoch_i,:,:,:]  = forward_convolution(sim_gt_full[epoch_i,:,:,:])




view_slices_3d(sim_gt_full[2,:,:,:], slice_nbr=50, vmin=-1, vmax=1, title="images of the susceptibility (ground truth)" )
print("view phase -conv mit dipole")
view_slices_3d(sim_fw_full[2,:,:,:], slice_nbr=50, vmin=-1, vmax=1, title= "conv of gt susc sources with dipole kernel")


###############################################################################
##############################################################################


#

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
#model.compile(optimizer='adam', loss='mean_absolute_error')
#model.summary()

###############################################
###############################################
print("Model with gradient accumulation")
gaaccumsteps = 8;
model = GradientAccumulateModel(accum_steps=gaaccumsteps, inputs=model.input, outputs=model.output)

# HERE ARE THE CHANGES!!!!!

model.compile(optimizer='adam', loss='mse') #mean_absolute_error
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



view_slices_3d(sim_fw_full[test_epoch_nbr, :, :, :], slice_nbr=16, vmin=-1, vmax=1, title='Input Tissue Phase')
view_slices_3d(sim_gt_full[test_epoch_nbr, :, :, :], slice_nbr=16, vmin=-1, vmax=1, title='GT Susceptibility')
view_slices_3d(y_pred[0, :, :, :, 0], slice_nbr=16, vmin=-1, vmax=1, title='Predicted Susceptibility')

################################################################################
# training part
##########################################################################



epochs_train = 60
save_period = 10
batch_size = 2
###############

# train
checkpoint_path = "checkpoints/GAcp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=False,
                                                 save_freq=save_period,
                                                 monitor = "val_loss",
                                                 verbose=1)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)




train_images =tf.expand_dims(sim_fw_full, 4)
train_labels = tf.expand_dims(sim_gt_full, 4)

print("fit model")
history = model.fit(train_images, train_labels,  epochs=epochs_train, batch_size=batch_size, shuffle=True,
          callbacks = [cp_callback])  # pass callback to training for saving the model80

loss_historyGA = history.history['loss']


with open('loss_historyGA.pickle', 'wb') as f:
    pickle.dump([loss_historyGA, epochs_train], f)

###################

#save model
if not os.path.exists("models"): 
    os.makedirs("models") 
    
model_name = "models/modelGA_" + str(num_train) +"trainsamples_"+ str(epochs_train) +"epochs_" + "batchsize"+ str(batch_size)+ "_"+ str(gaaccumsteps) +"gaaccum.h5"
#model.save(model_name)
tf.keras.models.save_model(model, model_name)    
    
    
###################
plt.plot(loss_historyGA)
plt.title("Loss")
plt.xlabel("Epochs ")
lossnamefile = "models/LOSSmodelGA_" + str(num_train) +"trainsamples_"+ str(epochs_train) +"epochs_" + "batchsize"+ str(batch_size)+ "_"+ str(gaaccumsteps) +"gaaccum.png"
plt.savefig(lossnamefile)
#################################################################################################
##################################################################################################

from  visualizenetworkprediction import visualize_all
## create test set

num_epochs = 3
size = 128

#

test_sim_gt_full = np.zeros((num_epochs, size, size, size)) 

test_sim_fw_full = np.zeros((num_epochs, size, size, size)) 

print("iterate epochs,sim susc,  convolve")
for epoch_i in range(num_epochs):
    if epoch_i ==1:
        plotH=True
    else:
         plotH=False
                   
    test_sim_gt_full[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = rect_num, plot=False)
    # forward convolution with the dipole kernel 
    test_sim_fw_full[epoch_i,:,:,:]  = forward_convolution(test_sim_gt_full[epoch_i,:,:,:])
#############################



print("trained")
# what does the untrained model predict

for epoch_i in range(num_epochs):
    X_test = test_sim_fw_full[np.newaxis, epoch_i,:,:,:, np.newaxis]

    y_pred = model.predict(X_test)

    print(epoch_i)
    title =   "trained network with testing data 154 epochs " + str(epoch_i)
    visualize_all(test_sim_fw_full[epoch_i,:,:,:], test_sim_gt_full[epoch_i,:,:,:], y_pred[0,:,:,:,0] ,title = title , save = True )


