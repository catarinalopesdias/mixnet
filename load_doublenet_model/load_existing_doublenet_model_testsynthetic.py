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
from plotting.visualize_volumes import visualize_all4


###############################################################################
# Load existing model
###############################################################################


# parameter models
num_train_instances = 115
dataset_iterations = 5000
#finalcheckpoint = 2996
batch_size = 1
gaaccumsteps = 10
num_filter = 16
losses = ["mse", "mse"]
text_stop = "stopping"
lr =0.003
text_lr = str(lr).split(".")[1]


#path = "checkpoints/doublenet/_BollmannPhillip_newadam" + str(num_filter)+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses[0] +losses[1]+text_stop+"_"+text_lr+".ckpt"
#path = "checkpoints/doublenet/newadam16_trainsamples150_datasetiter5000_batchsize1_gaaccum10_loss_msemse_stop3700_0003andstop.ckpt"
#"checkpoints/doublenet/newadam"+str(num_filter)+"_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+".ckpt"
#path = "checkpoints/doublenet/newadam"+str(num_filter)+"cp-"+ str(finalcheckpoint) +"_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+".ckpt"
#path = "checkpoints/GAcp-0"+ str(epochs_train) + str(num_train)+ "trainsamples_" + str(epochs_train) +"epochs_" + "batchsize"+ str(batch_size)+ "_"+ str(gaaccumsteps) +"gaaccum" + "loss"+str(lossU)+".ckpt"
#path = "checkpoints/GAcp-00"+str(num_epochs)+".ckpt/"
#path = "models/backgroundremovalBOLLMAN/modelBR_trainsamples100_datasetiter300_batchsize1_gaaccum10_loss_mse.keras"
#model = tf.keras.saving.load_model(path)


# Load model
#model.load_weights(model_file)
path = "checkpoints/doublenet/_BollmannPhillip_newadam" + str(num_filter)+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses[0] +losses[1]+text_stop+"_"+text_lr+"_valloss.ckpt"

model = tf.keras.models.load_model(path)

model.compile(loss = losses[0], optimizer = 'adam')



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
    text_typedata = "testdata"
else: #traindata
    loaded = np.load('datasynthetic/115samples.npz')
    text_typedata = "traindata"


phase = loaded['phase1']
phasebg = loaded['phase_bg1']
phase = loaded['phase1']
gt = loaded['sim_gt1']
del loaded

          
#######################################################
########################################################

if not os.path.exists("models/doublenet/prediction_images"): 
    os.makedirs("models/doublenet/prediction_images") 


for epoch_i in range(3): #num_instance
    X_test = phasebg[np.newaxis, epoch_i,:,:,:, np.newaxis]

    y_pred = model.predict(X_test)

    print(epoch_i)
    title =   "U1 network - Backgroundremoval: " + str(epoch_i)+ " " + losses[0] + " " +  text_typedata
    pathi =  "models/doublenet/prediction_images/model_BollmannPhillip_newadam_U1_" + str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1]+text_stop+"_"+text_lr +  text_typedata + "_epoch" + str(epoch_i) 
    predicted, reference,error = visualize_all4(phasebg[epoch_i,:,:,:], phase[epoch_i,:,:,:], y_pred[0][0,:,:,:,0] ,title = title , save = True, path = pathi )
    
    #############################################
    title =   "U2 -Total - Backgroud+Dipole: " + str(epoch_i)+ " " + losses[0] + " " +  text_typedata
    pathi =  "models/doublenet/prediction_images/model_BollmannPhillip_newadam_U2_" + str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1]+text_stop+"_"+text_lr +  text_typedata + "_epoch" + str(epoch_i) 
    predicted, reference,error = visualize_all4(phasebg[epoch_i,:,:,:], gt[epoch_i,:,:,:], y_pred[1][0,:,:,:,0] ,title = title , save = True, path = pathi )
    
