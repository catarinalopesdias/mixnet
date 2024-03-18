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
#from datahandling import read_and_code_tf
from plotting.visualize_volumes import visualize_all4
#from datahandlinging import visualize_volumes
#import read_and_code_tf


###############################################################################
# Load existing model
###############################################################################

# model data
#num_epochs = 60
#batch_size = 1

# model data


num_train_instances = 150
dataset_iterations = 5000
finalcheckpoint = 500
batch_size = 1
gaaccumsteps = 10
num_filter = 16


lossU = "mse" # "mean_absolute_error" #"mse"# "mean_absolute_error" #"mse"    #mse
#model_newadam_16filters_trainsamples150_datasetiter5000_batchsize1_gaaccum10_loss_mse_phillipp
#newadam_16filter_trainsamples150_datasetiter5000_batchsize1_gaaccum10_loss_mse_phillipp.ckpt
path = "checkpoints/dipoleinversion/newadam_"+str(num_filter)+"_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+"_phillipp.ckpt"
path = "checkpoints/dipoleinversion/newadam_16filter_trainsamples150_datasetiter5000_batchsize1_gaaccum10_loss_mse_phillipp.ckpt"
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

for epoch_i in range(3): #num_instance
   file =str(epoch_i)+"samples"

   if newdata:
        file =str(epoch_i)+"samples"
        #loaded = np.load(fullfile)
        text = "testdata"
   else: #traindata
        text = "traindata"
    
        file_full = "datasynthetic/npz/" + file + ".npz"

   loaded = np.load(file_full)
   loaded =loaded['arr_0']
   phase = loaded[1,:]
   gt = loaded[0,:]
   X_test = phase[np.newaxis, :,:,:, np.newaxis]

   y_pred = model.predict(X_test)

   print(epoch_i)
   title =   "trained network with testing data 50 epochs: " + str(epoch_i)+ " " + lossU
   pathi = "models/dipoleinversion/results/train"+str(num_filter) + "trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+ "_testset_epoch" + str(epoch_i) + text
   predicted, reference,error = visualize_all4(phase[:,:,:], gt[:,:,:], y_pred[0,:,:,:,0] ,title = title , save = True, path = pathi )
#########################

