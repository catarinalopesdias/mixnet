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
from plotting.visualize_volumes import visualize_all4, visualize_all4grey


#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################
newdata=False
name = "Bollmann"


path_common_init = "models/backgroundremovalBOLLMAN/prediction_images/unif/BgRem_"+name+"_newadam"


for epoch_i in range(1): #num_instance
   file =str(epoch_i)+"samples"
   
   print(file)

   if newdata:
        file =str(epoch_i)+"samples"
        #loaded = np.load(fullfile)
        text_typedata = "testdata"
        file_full = "datasynthetic/normal01evenlessbglessartifacts/npz/testing/" + file + ".npz"

   else: #traindata
        text_typedata = "traindata" 
        file_full = "datasynthetic//normal01evenlessbglessartifacts/npz/" + file + ".npz"

   loaded = np.load(file_full)
   loaded =loaded['arr_0']
   gt = loaded[0,:]
   phase = loaded[1,:]
   phasebg = loaded[2,:]
   #X_test = phasebg[np.newaxis, :,:,:, np.newaxis]

   #y_pred = model.predict(X_test)
   import  matplotlib.pyplot as plt

   plt.imshow(gt[64,:,:], cmap='gray',  vmin=-1.4, vmax=1.4)   
   plt.colorbar()
   max_o = str(round(np.max(gt),2))
   min_o = str(round(np.min(gt),2))

   #plt.imshow(y_pred[0,64,:,:,0], cmap='gray',  vmin=-0.01, vmax=0.01)   
   plt.title("file " + str(epoch_i) + ': gt - max: ' + max_o + ' min ' + min_o)
   plt.show()

   plt.imshow(phase[64,:,:], cmap='gray',  vmin=-1.4, vmax=1.4)  
   plt.colorbar()
   max_o = str(round(np.max(phase),2))
   min_o = str(round(np.min(phase),2))
   plt.title("file " +  str(epoch_i) + ': phase - max: ' + max_o + ' min ' + min_o)  
   plt.show()
   plt.imshow(phasebg[64,:,:], cmap='gray',  vmin=-1.4, vmax=1.4)   
   plt.colorbar()
   max_o = str(round(np.max(phasebg),2))
   min_o = str(round(np.min(phasebg),2))
   plt.title("file " +  str(epoch_i) + ': phasebg - max: '+ max_o + ' min ' + min_o)
   plt.show

                  
                  




  