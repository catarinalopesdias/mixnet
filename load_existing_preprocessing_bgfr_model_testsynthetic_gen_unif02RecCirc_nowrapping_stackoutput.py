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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
#from datahandling import read_and_decode_tf
from plotting.visualize_volumes import visualize_all4, visualize_all4grey
from plotting.visualize_volumes import view_slices_3d, view_slices_3dNew

import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle
from keras.models import Model
from keras.layers import Input
###############################################################################
# Load existing model
###############################################################################


# model data


#checkpoint path
#path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_Bgfrem_Bollmann_newadam16cp-3000_trainsamples500_datasetiter3000_batchsize1_gaaccum10_loss_costum_0025_val_loss_unif02_RecCirc__datagenRecCircNewLoss.ckpt"
#path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_Bgfrem_cat_newadam16cp-0065_trainsamples500_datasetiter3000_batchsize1_gaaccum10_loss_costum_0001_val_loss_unif02_RecCirc__datagenRecCircNewLoss.ckpt"
#path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_Bgfrem_cat_newadam16cp-0020_trainsamples500_datasetiter3000_batchsize1_gaaccum10_loss_costum_0001_val_loss_unif02_RecCirc__datagenRecCircNewLossOnlyBoundArtif.ckpt"
#path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_Bgfrem_cat_newadam16cp-0020_trainsamples500_datasetiter3000_batchsize1_gaaccum10_loss_costum_0001_val_loss_unif02_RecCirc__datagenRecCircNewLossOnlyBoundArtif.ckpt"
#path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_Bgfrem_cat_newadam16cp-2965_trainsamples500_datasetiter3000_batchsize1_gaaccum10_loss_costum_0001_val_loss_unif02_RecCirc__datagenRecCircNewLossOnlyBoundArtif.ckpt"

#path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_Bgfrem_cat4convs3levels_newadam16cp-2625_trainsamples500_datasetiter3000_batchsize1_gaaccum10_loss_costum_0001_val_loss_unif02_RecCirc__datagenRecCircNewLossOnlyBoundArtifOnlyBoundArtif.ckpt"
#path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_BgfRem4convs4levels_BgfRemov_newadam16cp-0571_trainsamples500_datasetiter3000_batchsize1_gaaccum10_loss_costum_0001_val_loss_unif02_RecCirc__phasemaskbgf_bgfremovalheber_datagenunif02rectcircles_wrap.ckpt"
#path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_BgfRem4convs4levels_BgfRemov_newadam16cp-0304_trainsamples1000_datasetiter3000_batchsize1_gaaccum10_loss_costum_0001_val_loss_unif02_RecCirc__phasemaskbgf_bgfremovalheber_datagenunif02rectcircles_wrap.ckpt"
path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_BgfRem4convs4levels_BgfRemovzgradient_no_boundartif_yes_wrap_yes__newadam16cp-0905_trainsamples1000_datasetiter3000_batchsize1_gaaccum10_loss_costum_0001_val_loss_unif02_RecCirc__phasemaskbgf_bgfremovalheber_datagenunif02rectcircles_wrap.ckpt"
model1= load_model(path, compile=False)
model1.summary()

#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################
newdata=False
import  matplotlib.pyplot as plt



path_common_init = "models/preprocessing_backgroundremoval/prediction_images/unif"

network_type = "PhaseMaskBgf_Bgfrem_Heber_newadam16"

for epoch_i in range(2): #num_instance
   #epoch_i=3
   file =str(epoch_i)+"samples"

   if newdata:
        file =str(epoch_i)+"samples"
        #loaded = np.load(fullfile)
        text_typedata = "testdata"
        #file_full = "datasynthetic/uniform02RectCircle-Phase/testing/" + file + ".npz"
        file_full = "datasynthetic/uniform02RectCircle-gt/testing/" + file + ".npz"


   else: #traindata
        text_typedata = "traindata" 
        #file_full = "datasynthetic/uniform02RectCircle-gt/training/" + file + ".npz"
        file_full = "datasynthetic/uniform02RectCircle-gt/training/" + file + ".npz"

   loaded = np.load(file_full)
   loaded =loaded['arr_0']

   gt = loaded
   
   
   X_test =  gt[np.newaxis, :,:,:, np.newaxis]

   y_pred = model1.predict(X_test)
   

   phaswBgF = y_pred[0][0,:,:,:,0] #1 output is bgf+phase 


   maskedphase  = y_pred[1][0,:,:,:,0] # 
   pred_phase = y_pred[1][0,:,:,:,1]

   print("max phase with bgf", phaswBgF.max() )
   print("min phase with bgf", phaswBgF.min() )
   
   


###############################################################################
###############################################################################

   title_t = "epoch " + str(epoch_i)
   predicted, reference,error = visualize_all4grey(phaswBgF, maskedphase, pred_phase ,
                                                   title = title_t, save = True,
                                                   path = "models/preprocessing_backgroundremoval/prediction_images/synthetic/newdata_"+str(newdata)+ "_epoch"+str(epoch_i)+"_gray",
                                                   colormax=0.25,colormin=-0.25,
                                                   errormax = 0.25,errormin=-0.25, slice_nr=64 )



   m_predicted, reference,error = visualize_all4(phaswBgF, maskedphase, pred_phase ,
                                                  title = title_t, save=True, 
                                                  path="models/preprocessing_backgroundremoval/prediction_images/synthetic/newdata_"+str(newdata)+ "_epoch"+str(epoch_i)+"color",
                                                   colormax=0.25,colormin=-0.25,
                                                   errormax = 0.1,errormin=-0.1  )




#############################################
########## Create new model  for real data
#############################################

import tensorflow as tf
from keras.layers import Input
from keras.models import Model
#from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
from networks.network_Heber_new4convs4levels import build_CNN_Heber_inputoutput
#from my_classes.keraslayer.layer_mask_inputtensor import CreateMaskLayer
#from my_classes.keraslayer.layerbackgroundfield_artifacts_nowrapping_nodim import CreatebackgroundFieldLayer
from newGA import GradientAccumulateModel
from keras.optimizers import Adam
import numpy as np


input_shape = (160, 160, 160, 1) # shape of pict, last is the channel

input_tensor = Input(shape = input_shape, name="input")


y = build_CNN_Heber_inputoutput(input_tensor)


model160 = Model(input_tensor, y)

model160.summary()


checkpoint = tf.train.Checkpoint(model160)
status = checkpoint.restore(path) 
#status.expect_partial()



#################################################



# load unseen data 
################################################
#   Import data
################################################

from datahandling import read_and_decode_tf
import nibabel as nib


path_common_init = "models/preprocessing_backgroundremoval/prediction_images/real/"
network_type = "BgRem_Heber_newadam"
##############################################################
###############################################################################
########## load nii.gz data
###############################################################################

folder_dir ="datareal/qsm2016_recon_challenge/data/"
file = "msk.nii.gz"
filetoupload = folder_dir + file



img = nib.load(filetoupload)
###########################################


####################
# load phase tissue
###################
file = "phs_tissue.nii.gz"
filetoupload = folder_dir + file
img = nib.load(filetoupload)
reference_phasetissue = img.get_fdata()
reference_phasetissue = reference_phasetissue[np.newaxis, :,:,:, np.newaxis]

###################
#load wrap data 
###################
file = "phs_wrap.nii.gz"
filetoupload = folder_dir + file
img = nib.load(filetoupload)
input_phasebgwrap = img.get_fdata()
input_phasebgwrap = input_phasebgwrap[np.newaxis, :,:,:, np.newaxis]


###################
# load unwrap data
###################
file = "phs_unwrap.nii.gz"
filetoupload = folder_dir + file
img = nib.load(filetoupload)
input_phasebgunwrap = img.get_fdata()
input_phasebgunwrap = input_phasebgunwrap[np.newaxis, :,:,:, np.newaxis]

###################
# load mask data
###################

#loas mask
file = "msk.nii.gz"
filetoupload = folder_dir + file
img = nib.load(filetoupload)
mask = img.get_fdata()
########################################





predictedPhase = model160.predict(input_phasebgunwrap)

prediction_title = "nii.gz files  - qsm2016"
pathi = path_common_init + network_type  + "niigz_nowrapping"



m_predicted, reference,error = visualize_all4(input_phasebgunwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0],
                                                  title = prediction_title, save=True, 
                                                  path=pathi,
                                                   colormax=0.05,colormin=-0.05,
                                                   errormax = 0.1,errormin=-0.1  )

pathi = path_common_init + network_type  + "niigz_nowrapping_gray"

predicted, reference,error = visualize_all4grey(input_phasebgunwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0] ,
                                                   title = "ff" , save = True,
                                                   path = pathi,
                                                   colormax=0.05,colormin=-0.05,
                                                   errormax = 0.13,errormin=-0.13, slice_nr=64 )


#predicted, reference,error = visualize_all4grey(input_phasebgunwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0] ,
#                                                   title = "ff" , save = True,
#                                                   path = pathi,
#                                                   colormax=0.05,colormin=-0.05,
#                                                   errormax = 0.13,errormin=-0.13, slice_nr=100 )
#   