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
from plotting.visualize_volumes import visualize_all4, visualize_all4grey


###############################################################################
# Load existing model
###############################################################################

# model data
#num_epochs = 60
#batch_size = 1

# model datalossss
num_train_instances = 150
dataset_iterations = 5000
finalcheckpoint = 3542
batch_size = 1
gaaccumsteps = 10
num_filter = 16


lossU = "mse" 

#newadam16cp-3542_trainsamples150_datasetiter5000_batchsize1_gaaccum10_loss_mse.ckpt
path = "checkpoints/bgremovalmodel/newadam"+str(num_filter)+"cp-"+ str(finalcheckpoint) + \
    "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
        "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+".ckpt"


model = tf.keras.models.load_model(path)
#model = tf.keras.saving.load_model(path)


model.compile(loss = lossU, optimizer = 'adam')



#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################
newdata=False

# compressed data
if newdata:
    loaded = np.load('datasynthetic/5samples.npz')
    text = "testdata"
else: #traindata
    loaded = np.load('datasynthetic/150samples.npz')
    text = "traindata"


phase = loaded['phase1']
phasebg = loaded['phase_bg1']
del loaded
#new data
#phasebg  = np.load('datasynthetic/50phase_bg.npy')
#phase  = np.load('datasynthetic/50phase.npy')
#phasebg  = np.load('datasynthetic/phase_bg100.npy')
#phase  = np.load('datasynthetic/phase100.npy')
num_instance = 50
          





for epoch_i in range(2): #num_instance
    X_test = phasebg[np.newaxis, epoch_i,:,:,:, np.newaxis]

    y_pred = model.predict(X_test)

    print(epoch_i)
    title =   "trained network with testing data 50 epochs: " + str(epoch_i)+ " " + lossU
    pathi = "models/backgroundremovalBOLLMAN/results/train"+str(num_filter) + "trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+ "_testset_epoch" + str(epoch_i) + text
    predicted, reference,error = visualize_all4(phasebg[epoch_i,:,:,:], phase[epoch_i,:,:,:], y_pred[0,:,:,:,0] ,title = title , save = False, path = pathi )
    
    
    predicted, reference,error = visualize_all4grey(phasebg[epoch_i,:,:,:], phase[epoch_i,:,:,:], y_pred[0,:,:,:,0] ,
                                                    title = title , save = False,
                                                    path = pathi,
                                                    colormax=0.4,colormin=-0.4,
                                                    errormax = 0.4,errormin=-0.4 )
    
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