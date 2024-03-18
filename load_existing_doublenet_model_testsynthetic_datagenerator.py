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
#from datahandling import read_and_decode_tf
from plotting.visualize_volumes import visualize_all4


###############################################################################
# Load existing model
###############################################################################


# parameter models
num_train_instances = 500
dataset_iterations = 5000
#finalcheckpoint = 2996
batch_size = 1
gaaccumsteps = 10
num_filter = 16
losses = ["mse", "mse"]
text_stop = "stopping"
lr =0.003
text_lr = str(lr).split(".")[1]

###############################################################################

# Load model
#model.load_weights(model_file)
path = "checkpoints/doublenet/_BollmannPhillip_newadam" + \
        str(num_filter)+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
            "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses[0] +losses[1]+ \
                text_stop+"_"+text_lr+"_valloss.ckpt"


model = tf.keras.models.load_model(path)

model.compile(loss = losses[0], optimizer = 'adam')



#################################################################
#################################################################

if not os.path.exists("models/doublenet/prediction_images"): 
    os.makedirs("models/doublenet/prediction_images") 


# load unseen data 
################################################
#   Import data
################################################
path_common_init = "models/doublenet/prediction_images/model_BollmannPhillip_newadam"

newdata=True

for epoch_i in range(3): #num_instance
   file =str(epoch_i)+"samples"

   if newdata:
        file =str(epoch_i)+"samples"
        #loaded = np.load(fullfile)
        text_typedata = "testdata"
        file_full = "datasynthetic/npz/testing/" + file + ".npz"

   else: #traindata
        text_typedata = "traindata"
    
        file_full = "datasynthetic/npz/" + file + ".npz"

###############################################
   loaded = np.load(file_full)
   loaded =loaded['arr_0']
   phasebg = loaded[2,:]
   phase = loaded[1,:]
   gt = loaded[0,:]
   X_test = phasebg[np.newaxis, :,:,:, np.newaxis]

   y_pred = model.predict(X_test)

   print(epoch_i)
   #title =   "trained network with testing data 50 epochs: " + str(epoch_i)+ " " + lossU
   #pathi = "models/dipoleinversion/results/train"+str(num_filter) + "trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+ "_testset_epoch" + str(epoch_i) + text
   #predicted, reference,error = visualize_all4(phase[:,:,:], gt[:,:,:], y_pred[0,:,:,:,0] ,title = title , save = True, path = pathi )
   
   path_common_final =  str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + \
               "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1]+text_stop+"_"+text_lr +  text_typedata + "_epoch" + str(epoch_i) + "valloss"

   print(epoch_i)
   title =   "U1 network - Backgroundremoval: " + str(epoch_i)+ " " + losses[0] + " " +  text_typedata
   pathi =  path_common_init + "_U1_" + path_common_final
       
   predicted, reference,error = visualize_all4(phasebg[:,:,:], phase[:,:,:], y_pred[0][0,:,:,:,0] ,title = title , save = True, path = pathi )
    
    #############################################
   title =   "U2 -Total - Backgroud+Dipole: " + str(epoch_i)+ " " + losses[0] + " " +  text_typedata
   pathi =  path_common_init +  "_U2_" + path_common_final

   predicted, reference,error = visualize_all4(phasebg[:,:,:], gt[:,:,:], y_pred[1][0,:,:,:,0] ,title = title , save = True, path = pathi )
    
    
    
    



          
#######################################################
########################################################


"""

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
    """
