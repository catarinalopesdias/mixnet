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
text_susc="unif02"

#DipInv_Bollmann_newadam16_trainsamples500_datasetiter5000_batchsize1_gaaccum10_loss_mse_003_val_loss_datagen.ckpt

#DipInv_Phillip_newadam16_trainsamples500_datasetiter5000_batchsize1_gaaccum10_loss_mse_0005_val_loss_unif02_datagen.ckpt

name = "Phillipp" # Phillip
path = "checkpoints/dipoleinversion/DipInv_" + name + "_newadam" + \
        str(num_filter)+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
            "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses + "_" + text_lr \
              + "_" + "val_loss"+ "_"+ text_susc +"_datagen" + ".ckpt"




model = tf.keras.models.load_model(path)
#model = tf.keras.saving.load_model(path)


model.compile(loss = losses, optimizer = 'adam')



#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################
newdata=False



path_common_init = "models/dipoleinversion/prediction_images/unif/DipInv_"+name+"_newadam"


for epoch_i in range(3): #num_instance
    
   file =str(epoch_i)+"samples"

   if newdata:
        file =str(epoch_i)+"samples"
        #loaded = np.load(fullfile)
        text_typedata = "testdata"
        file_full = "datasynthetic/uniform02/npz/testing/" + file + ".npz"

   else: #traindata
        text_typedata = "traindata" 
        file_full = "datasynthetic//uniform02/npz/" + file + ".npz"

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
   import  matplotlib.pyplot as plt
   plt.imshow(phase[64,:,:], cmap='gray',  vmin=-0.4, vmax=0.4)   


   predicted, reference,error = visualize_all4grey(phase[:,:,:], gt[:,:,:], y_pred[0,:,:,:,0] ,
                                                   title = title , save = True,
                                                   path = pathi,
                                                   colormax=0.4,colormin=-0.4,
                                                   errormax = 0.4,errormin=-0.4 )
   
#########################
# 2d dif
dim = int(gt.shape[0]/2)
gt2d=gt[dim,:,:]
pred2d= y_pred[0,dim,:,:,0]

diff2d = pred2d-gt2d

#plot 2d diff
import matplotlib.pyplot as plt
import scipy.ndimage
plt.imshow(diff2d, cmap='RdBu',  vmin=-0.2, vmax=0.2)
plt.show()
###############
#3d diff

gt3d=gt[:,:,:]
pred3d= y_pred[0,:,:,:,0]


diff3d = pred3d-gt3d
flatten_diff3d =np.ndarray.flatten(diff3d)

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

sns.kdeplot(flatten_diff3d, label="Seaborn KDE Implementation",
            fill =True, common_norm= True)  #bw_adjust=1
from scipy.stats import norm 
norm.fit(flatten_diff3d)
print("mean", norm.fit(flatten_diff3d)[0])
print("std", norm.fit(flatten_diff3d)[1])
#plt.imshow(bla1, cmap='RdBu',  vmin=-1.5, vmax=1.5)
#plt.imshow(bla2, cmap='RdBu',  vmin=-1.5, vmax=1.5)
#diff = bla1-bla2
#plt.imshow(bla1-bla2, cmap='RdBu',  vmin=-1.5, vmax=1.5)
#plt.colorbar()
#import sklearn
#from sklearn.neighbors import KernelDensity
###########################
# svmbir (super-voxel model-based iterative reconstruction) is an easy-to-use python package for fast iterative
def nrmse(image, reference_image):
    """
    Compute the normalized root mean square error between image and reference_image.

    Args:
        image: Calculated image
        reference_image: Ground truth image

    Returns:
        Root mean square of (image - reference_image) divided by RMS of reference_image

    """
    rmse = np.sqrt(((image - reference_image) ** 2).mean())
    denominator = np.sqrt(((reference_image) ** 2).mean())

    return rmse/denominator


nrmse(pred3d, gt3d)

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
    
compare2d(pred2d, gt2d)
nrmse(pred3d, gt3d,normalization="min-max")
nrmse(pred3d, gt3d,normalization="euclidean")
nrmse(pred3d, gt3d,normalization="mean")

nrmse(pred2d, gt2d,normalization="min-max")
nrmse(pred2d, gt2d,normalization="euclidean")
nrmse(pred2d, gt2d,normalization="mean")


###########

# Helper function for plotting histograms.
def plot_hist(ax, data, title=None):
    ax.hist(data.ravel(), bins=256)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    
    if title:
        ax.set_title(title)
        
fig, ((a, b), (c, d)) = plt.subplots(nrows=2, ncols=2)

plot_hist(a, diff3d, title="Original")
plt.show()
