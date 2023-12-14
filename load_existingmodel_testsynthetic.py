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
num_epochs = 50
batch_size = 1



path = "checkpoints/GAcp-00"+str(num_epochs)+".ckpt/"
model = tf.keras.models.load_model(path)


model.compile(loss = 'mean_squared_error', optimizer = 'adam')



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
# what does the untrained model predict

for epoch_i in range(num_epochs):
    X_test = test_sim_fw_full[np.newaxis, epoch_i,:,:,:, np.newaxis]

    y_pred = model.predict(X_test)

    print(epoch_i)
    title =   "trained network with testing data 46 epochs " + str(epoch_i)
    a, b,e = visualize_all(test_sim_fw_full[epoch_i,:,:,:], test_sim_gt_full[epoch_i,:,:,:], y_pred[0,:,:,:,0] ,title = title , save = True )


import matplotlib.pyplot as plt

nn = np.array( b)
bb = nn[0,1:50,50:100]

#
bla =test_sim_gt_full[1,5,:,:]
#bla = y_pred[0,5,:,:,0]
plt.matshow(bla)
plt.colorbar()
plt.clim(-1, 1);

plt.show()