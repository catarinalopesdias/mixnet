#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:24:18 2023

@author: catarinalopesdias
"""

"""#Created on Mon Nov 20 10:03:11 2023

@author: catarinalopesdias
"""

# load and evaluate a saved model
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
from datahandling import read_and_decode_tf
from visualizenetworkprediction import visualize_all
from create_datasetfunctions import simulate_susceptibility_sources, generate_3d_dipole_kernel, forward_convolution
from tensorflow import keras

###############################################################################
# Load existing model
###############################################################################

# model data
#num_epochs = 60
#batch_size = 1

# model data


num_train_instances = 100
dataset_iterations = 100
batch_size = 2
gaaccumsteps = 8;
lossU = "mean_absolute_error" #"mse"# "mean_absolute_error" #"mse"    #mse

path = "checkpoints/bgremovalmodel/cp-{epoch:04d}"+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+".ckpt"
#path = "checkpoints/GAcp-0"+ str(epochs_train) + str(num_train)+ "trainsamples_" + str(epochs_train) +"epochs_" + "batchsize"+ str(batch_size)+ "_"+ str(gaaccumsteps) +"gaaccum" + "loss"+str(lossU)+".ckpt"
#path = "checkpoints/GAcp-00"+str(num_epochs)+".ckpt/"
model = tf.keras.models.load_model(path)


model.compile(loss = lossU, optimizer = 'adam')



#################################################################
#################################################################


num_epochs = 2
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
                   
    test_sim_gt_full[epoch_i,:,:,:] = simulate_susceptibility_sources(simulation_dim = size, rectangles_total = 40, plot=False)
    # forward convolution with the dipole kernel 
    test_sim_fw_full[epoch_i,:,:,:]  = forward_convolution(test_sim_gt_full[epoch_i,:,:,:])
#############################



print("trained")

for epoch_i in range(num_epochs):
    X_test = test_sim_fw_full[np.newaxis, epoch_i,:,:,:, np.newaxis]

    y_pred = model.predict(X_test)

    print(epoch_i)
    title =   "trained network with testing data 46 epochs " + str(epoch_i)
    predicted, reference,error = visualize_all(test_sim_fw_full[epoch_i,:,:,:], test_sim_gt_full[epoch_i,:,:,:], y_pred[0,:,:,:,0] ,title = title , save = True )
#########################

#import matplotlib.pyplot as plt

#ref_np = np.array( reference)
#ref_piece = ref_np[0,1:50,50:100]

#
#bla =reference[1]
#bla =predicted[1]

#plt.matshow(bla)
#plt.colorbar()
#plt.clim(-1, 1);

#plt.show()