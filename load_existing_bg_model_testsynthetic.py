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
from pl import visualize_all4


###############################################################################
# Load existing model
###############################################################################

# model data
#num_epochs = 60
#batch_size = 1

# model datalossss
num_train_instances = 150
dataset_iterations = 5000
finalcheckpoint = 2996
batch_size = 1
gaaccumsteps = 10
num_filter = 16


lossU = "mse" # "mean_absolute_error" #"mse"# "mean_absolute_error" #"mse"    #mse

#newadam16cp-2996_trainsamples150_datasetiter3000_batchsize1_gaaccum10_loss_mse.ckpt
#newadam16cp-1981_trainsamples150_datasetiter2000_batchsize1_gaaccum10_loss_mse.ckpt
#newadam16cp-1307_trainsamples150_datasetiter2000_batchsize1_gaaccum10_loss_mse.ckpt
#n#ewadam16cp-1977_trainsamples130_datasetiter2000_batchsize1_gaaccum10_loss_mse.ckpt
#newadam64cp-0997_trainsamples100_datasetiter999_batchsize1_gaaccum10_loss_mse.ckpt
#32cp-0416_trainsamples100_datasetiter999_batchsize1_gaaccum10_loss_mse.ckpt
#32cp-0469_trainsamples100_datasetiter999_batchsize1_gaaccum10_loss_mse.ckpt
#cp-0326_trainsamples100_datasetiter999_batchsize1_gaaccum10_loss_mse.ckpt
#cp-0300_trainsamples100_datasetiter300_batchsize1_gaaccum10_loss_mse.ckpt.index
#path = "checkpoints/bgremovalmodel/cp-0"+ str(finalcheckpoint) +"_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+".ckpt.index"
#model_newadam_16filters_trainsamples150_datasetiter5000_batchsize1_gaaccum10_loss_mse_phillipp
#model_newadam_16filters_trainsamples150_datasetiter5000_batchsize1_gaaccum10_loss_mse_phillip#
path = "checkpoints/bgremovalmodel/newadam"+str(num_filter)+"cp-"+ str(finalcheckpoint) +"_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+".ckpt"

#path = "checkpoints/GAcp-0"+ str(epochs_train) + str(num_train)+ "trainsamples_" + str(epochs_train) +"epochs_" + "batchsize"+ str(batch_size)+ "_"+ str(gaaccumsteps) +"gaaccum" + "loss"+str(lossU)+".ckpt"
#path = "checkpoints/GAcp-00"+str(num_epochs)+".ckpt/"

#path = "models/backgroundremovalBOLLMAN/modelBR_trainsamples100_datasetiter300_batchsize1_gaaccum10_loss_mse.keras"
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
    loaded = np.load('datasynthetic/10samples.npz')
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
          





for epoch_i in range(3): #num_instance
    X_test = phasebg[np.newaxis, epoch_i,:,:,:, np.newaxis]

    y_pred = model.predict(X_test)

    print(epoch_i)
    title =   "trained network with testing data 50 epochs: " + str(epoch_i)+ " " + lossU
    pathi = "models/backgroundremovalBOLLMAN/results/train"+str(num_filter) + "trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+ "_testset_epoch" + str(epoch_i) + text
    predicted, reference,error = visualize_all4(phasebg[epoch_i,:,:,:], phase[epoch_i,:,:,:], y_pred[0,:,:,:,0] ,title = title , save = True, path = pathi )
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