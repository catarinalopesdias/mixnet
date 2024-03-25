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


num_train_instances = 500
dataset_iterations = 5000
batch_size = 1
gaaccumsteps = 10
num_filter = 16
#text_stop = "stopping"
lr =0.0005
text_lr = str(lr).split(".")[1]


losses = "mse" # "mean_absolute_error" #"mse"

#DipInv_Bollmann_newadam16_trainsamples500_datasetiter5000_batchsize1_gaaccum10_loss_mse_003_val_loss_datagen.ckpt
name = "Phillipp" # Phillip
path = "checkpoints/dipoleinversion/DipInv_" + name + "_newadam" + \
        str(num_filter)+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
            "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses + "_" + text_lr \
              + "_" + "val_loss"+ "_datagen" + ".ckpt"



model = tf.keras.models.load_model(path)
#model = tf.keras.saving.load_model(path)


model.compile(loss = losses, optimizer = 'adam')



#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################
newdata=True




path_common_init = "models/dipoleinversion/prediction_images/DipInv_"+name+"_newadam"


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

   loaded = np.load(file_full)
   loaded =loaded['arr_0']
   phase = loaded[1,:]
   gt = loaded[0,:]
   X_test = phase[np.newaxis, :,:,:, np.newaxis]

   y_pred = model.predict(X_test)





   path_common_final =  str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
                  "_batchsize"+ str(batch_size) + "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses +"_"+text_lr +\
                      "_"  +  "valloss"+"_datagen_"+ text_typedata + "_epoch" + str(epoch_i) 
                  
                  
   print(epoch_i)
   title =   text_typedata + "_epoch " + str(epoch_i)+ " " + losses


   pathi =  path_common_init  + path_common_final



   predicted, reference,error = visualize_all4(phase[:,:,:], gt[:,:,:], y_pred[0,:,:,:,0] ,title = title , save = True, path = pathi )
   
#########################
"""
dim = int(gt.shape[0]/2)
bla1=gt[dim,:,:]
bla2= y_pred[0,dim,:,:,0]

diff = bla2-bla1
import matplotlib.pyplot as plt
import scipy.ndimage
plt.imshow(diff, cmap='RdBu',  vmin=-0.2, vmax=0.2)


plt.imshow(bla1, cmap='RdBu',  vmin=-1.5, vmax=1.5)
plt.imshow(bla2, cmap='RdBu',  vmin=-1.5, vmax=1.5)
diff = bla1-bla2
plt.imshow(bla1-bla2, cmap='RdBu',  vmin=-1.5, vmax=1.5)
plt.colorbar()
import sklearn
from sklearn.neighbors import KernelDensity

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(diff)
 kde.score_samples(diff)
 sns.kdeplot(data=diff, x="total_bill")
 
matplotlib.pyplot.hist( scipy.stats.gaussian_kde(diff, bw_method=None, weights=None))
import matplotlib
matplotlib.pyplot.hist(np.ndarray.flatten(diff), bins=128)"""