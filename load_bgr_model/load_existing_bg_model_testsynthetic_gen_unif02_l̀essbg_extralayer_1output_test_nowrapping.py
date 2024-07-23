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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
#from datahandling import read_and_decode_tf
from plotting.visualize_volumes import visualize_all4, visualize_all4grey


###############################################################################
# Load existing model
###############################################################################


# model data


num_train_instances = 500
dataset_iterations = 2000
batch_size = 1
gaaccumsteps = 10
num_filter = 16
#text_stop = "stopping"
lr =0.001
text_lr = str(lr).split(".")[1]


losses = "mse" # "mean_absolute_error" #"mse"
text_susc="unif02"

name = "BollmannExtralayer" # Phillip
lastit="0001"
#Bg_Bollmann_newadam16cp-1589_trainsamples500_datasetiter3000_batchsize1_gaaccum10_loss_mse_0005_val_loss_unif02_datagen_lessbg.ckpt
#Bg_Bollmann_newadam16cp-1181_trainsamples500_datasetiter2000_batchsize1_gaaccum10_loss_mse_0004_val_loss_unif02_datagen_lessbg.ckpt


#Bg_Bollmann_newadam16cp-1111_trainsamples500_datasetiter4000_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_datagen_evenlessbg.ckpt


#Bg_BollmannExtralayer_newadam16cp-1533_trainsamples500_datasetiter2000_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayer.ckpt
 #Bg_BollmannExtraLayer_newadam16cp-1533_trainsamples500_datasetiter2000_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayer.ckpt

#Bg_BollmannExtralayer_newadam16cp-0052_trainsamples500_datasetiter2000_batchsize1_gaaccum10_loss_mse_01_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayer.ckpt

path = "checkpoints/bgremovalmodel_ExtraLayer/Bg_" + name + "_newadam" + \
        str(num_filter)+"cp-"+ lastit+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
            "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses + "_" + text_lr \
              + "_" + "val_loss"+ "_"+ text_susc +"_datagen" + "_evenlessbgnoartifacts_ExtraLayer_artif_1_nowrapping1.ckpt"



model = tf.keras.models.load_model(path)
#model = tf.keras.saving.load_model(path)


#model.compile(loss = losses, optimizer = 'adam')

########################
# create new model
###################################
#from keras.layers import Input
from keras.models import Model
model2 = Model(inputs=model.inputs, outputs=[model.layers[2].output, model.outputs[0]])






#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################
newdata=True



path_common_init = "models/backgroundremovalBOLLMAN_ExtraLayer/prediction_images/unif"

network_type = "BgRem_"+name+"_newadam"

for epoch_i in range(10): #num_instance
   file =str(epoch_i)+"samples"

   if newdata:
        file =str(epoch_i)+"samples"
        #loaded = np.load(fullfile)
        text_typedata = "testdata"
        file_full = "datasynthetic/uniform02_Rect_mask_phase/npz/testing/" + file + ".npz"

   else: #traindata
        text_typedata = "traindata" 
        file_full = "datasynthetic//uniform02_Rect_mask_phase/npz/" + file + ".npz"

   loaded = np.load(file_full)
   loaded =loaded['arr_0']
   mask = loaded[0,:]
   phase = loaded[1,:]
   #phasebg = loaded[2,:]
   X_test = [mask[np.newaxis, :,:,:, np.newaxis], phase[np.newaxis, :,:,:, np.newaxis]]

   y_pred = model2.predict(X_test)
   
   pred_bgf = y_pred[0]
   pred_phase = y_pred[1]
   

   
   
   import  matplotlib.pyplot as plt

   #plt.imshow(X_test[0,64,:,:,0], cmap='gray',  vmin=-0.4, vmax=0.4)   
   #plt.imshow(y_pred[0,64,:,:,0], cmap='gray',  vmin=-0.01, vmax=0.01)   

   #plt.imshow(gt[64,:,:], cmap='gray',  vmin=-0.4, vmax=0.4)   
   #plt.imshow(phase[64,:,:], cmap='gray',  vmin=-0.4, vmax=0.4)   
   #plt.imshow(phasebg[64,:,:], cmap='gray',  vmin=-0.4, vmax=0.4)   



   path_common_final =  str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
                  "_batchsize"+ str(batch_size) + "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses +"_"+text_lr +\
                      "_"  +  "valloss"+"_datagen_"+ text_typedata +"_" + lastit+ "_epoch" + str(epoch_i) + "_evenlessbgttt_nowrapping"
                  
                  
   print(epoch_i)
   title =   text_typedata + "_epoch " + str(epoch_i)+ " " + losses


   pathi =  path_common_init + "/grey/"+ path_common_final



   predicted, reference,error = visualize_all4grey(pred_bgf[0,:,:,:,0], phase[:,:,:], pred_phase[0,:,:,:,0] ,
                                                   title = title , save = True,
                                                   path = pathi,
                                                   colormax=0.2,colormin=-0.2,
                                                   errormax = 0.2,errormin=-0.2 )
   
   
   pathi = path_common_init +"/color/"+ text_typedata + "/"+ network_type  + path_common_final+"_color"
   
   predicted, reference,error = visualize_all4( pred_bgf[0,:,:,:,0], phase[:,:,:], pred_phase[0,:,:,:,0],
                                                   title = title , save = True,
                                                   path = pathi,
                                                   colormax=0.2,colormin=-0.2,
                                                   errormax = 0.1,errormin=-0.1)
   
   
#########################
# 2d dif
dim = int(phase.shape[0]/2)
phase2d=phase[dim,:,:]
pred2d= y_pred[0,dim,:,:,0]

diff2d = pred2d-phase2d

#plot 2d diff
import matplotlib.pyplot as plt
import scipy.ndimage


plt.imshow(diff2d, cmap='RdBu',  vmin=-0.2, vmax=0.2)
plt.colorbar()
plt.title("difference prediction and phase")
plt.show()
print("max 2d phase ", np.max(phase2d))

################



#plt.imshow(phasebg[dim,:,:], cmap='RdBu',  vmin=-3, vmax=3 )
#plt.colorbar()
#plt.title("phase and background")
#plt.show()

plt.imshow(phase2d, cmap='RdBu',  vmin=-0.2, vmax=0.2 )
plt.colorbar()
plt.title("phase")
plt.show()


#plt.imshow(phase2d, cmap='gray',  vmin=-0.4, vmax=0.4 )
#plt.colorbar()
#plt.title("phase")
#plt.show()
#print("max 2d phase ", np.max(phase2d))


plt.imshow(pred2d, cmap='RdBu',  vmin=-0.2, vmax=0.2)
plt.colorbar()
plt.title("prediction")
plt.show()
print("max 2d prediction ", np.max(pred2d))


###############
#3d diff

phase3d=phase[:,:,:]
pred3d= y_pred[0,:,:,:,0]


diff3d = pred3d-phase3d
flatten_diff3d =np.ndarray.flatten(diff3d)

print("max difference3d",np.max(diff3d))
print("max phase3d", np.max(phase3d))
print("max pred3d", np.max(pred3d))


#########################


import seaborn as sns

#kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(diff)
#kde.score_samples(diff)
#sns.kdeplot(data=diff, x="total_bill")
 
#################################################

#histogram
plt.hist(flatten_diff3d, bins=2000, density = True)
plt.show()
############################################################

# i have no clue what does this do

#from sklearn.neighbors import KernelDensity

#flatten_diff_reshape = flatten_diff.reshape(-1, 1)
#kde = KernelDensity(bandwidth=0.5, kernel='gaussian').fit(flatten_diff_reshape)
#plt.plot(kde)
#log_density = kde.score_samples(flatten_diff_reshape)

#################################

#sns.kdeplot(flatten_diff3d, label="Seaborn KDE Implementation",
#            fill =True, common_norm= True)  #bw_adjust=1
#from scipy.stats import norm 
#norm.fit(flatten_diff3d)
#print("mean", norm.fit(flatten_diff3d)[0])
#print("std", norm.fit(flatten_diff3d)[1])
#plt.imshow(bla1, cmap='RdBu',  vmin=-1.5, vmax=1.5)
#plt.imshow(bla2, cmap='RdBu',  vmin=-1.5, vmax=1.5)
#diff = bla1-bla2
#plt.imshow(bla1-bla2, cmap='RdBu',  vmin=-1.5, vmax=1.5)
#plt.colorbar()
#import sklearn
#from sklearn.neighbors import KernelDensity
###########################
# svmbir (super-voxel model-based iterative reconstruction) is an easy-to-use python package for fast iterative
"""def nrmse(image, reference_image):
 
    Compute the normalized root mean square error between image and reference_image.

    Args:
        image: Calculated image
        reference_image: Ground truth image

    Returns:
        Root mean square of (image - reference_image) divided by RMS of reference_image


    rmse = np.sqrt(((image - reference_image) ** 2).mean())
    denominator = np.sqrt(((reference_image) ** 2).mean())

    return rmse/denominator


#nrmse(pred3d, phase3d)
"""
#############
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse

###########

######

def compare2d(imageA, imageB):
    # Calculate the MSE and SSIM
    max_all = min(imageA.max(), imageB.max())
    min_all  = min(imageA.min(), imageB.min())
    m = mse(imageA, imageB)
    nm = nrmse(imageA, imageB)

    s = ssim(imageA, imageB, data_range=0.4) #??
    print("2d mean square error", m) 
    print("2d normalized root mean square error", nm) 
    print("2d sdsim",1- s) 
    ####################

def compare3d(imageA, imageB):
    # Calculate the MSE and SSIM
    max_all = min(imageA.max(), imageB.max())
    min_all  = min(imageA.min(), imageB.min())
    m = mse(imageA, imageB)
    nm = nrmse(imageA, imageB)

   # s = ssim(imageA, imageB, data_range=[min_all, max_all])
    print("3d mean square error", m) 
    print("3d normalized root mean square error", nm) 

    #print("ssim", s) 
    
#compare2d(pred2d, phase2d)
#nrmse(pred3d, phase3d,normalization="min-max")
#nrmse(pred3d, phase3d,normalization="euclidean")
#nrmse(pred3d, phase3d,normalization="mean")

#nrmse(pred2d, phase2d,normalization="min-max")
#nrmse(pred2d, phase2d,normalization="euclidean")
#nrmse(pred2d, phase2d,normalization="mean")


###########

## Helper function for plotting histograms.
def plot_hist(ax, data, title=None):
    ax.hist(data.ravel(), bins=256)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    
    if title:
        ax.set_title(title)
        
#fig, ((a, b), (c, d)) = plt.subplots(nrows=2, ncols=2)

#plot_hist(a, diff3d, title="Original")

#########################################################################

#for epoch_i in range(3): #num_instance
 #  file =str(epoch_i)+"samples"

   #if newdata:
    #    file =str(epoch_i)+"samples"
     #   #loaded = np.load(fullfile)
      #  text_typedata = "testdata"
        #file_full = "datasynthetic/uniform02evenlessbglessartifacts/npz/testing/" + file + ".npz"

   #else: #traindata
    #    text_typedata = "traindata" 
   #file_full = "datasynthetic//uniform02evenlessbglessartifacts/npz/" + file + ".npz"

   #loaded = np.load(file_full)
   #loaded =loaded['arr_0']
   #gt = loaded[0,:]
   #phase = loaded[1,:]
   #phasebg = loaded[2,:]
   #X_test = phasebg[np.newaxis, :,:,:, np.newaxis]
   #plt.imshow(gt[64,:,:], cmap='RdBu',  vmin=-0.005, vmax=0.005)
   #plt.show()
   #y_pred = model.predict(X_test)
   #import  matplotlib.pyplot as plt
   #print(np.max(gt))
   #print(np.max(phase))
   #print(np.max(phasebg))#   
   #print(np.min(phasebg))
   
   
   
   
   