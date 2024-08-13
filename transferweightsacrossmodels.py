#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:22:31 2024
Load checkpoint
Create new model
load weights from other model 
@author: catarinalopesdias
"""
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
from my_classes.keraslayer.layerbackgroundfield_artifacts_nowrapping_nodim import CreatebackgroundFieldLayer
from newGA import GradientAccumulateModel
from keras.optimizers import Adam
import numpy as np
#############################################
########## Load checkpoint and put it in to model
#############################################


# model data
# 
"""num_train_instances = 500
dataset_iterations = 5000
batch_size = 1
gaaccumsteps = 10
num_filter = 16
#text_stop = "stopping"
lr =0.001
text_lr = str(lr).split(".")[1]


losses = "mse" # "mean_absolute_error" #"mse"
text_susc="unif02"

name = "BollmannExtralayer" # Phillip
lastit="0820"

#checkpoint path
path = "checkpoints/bgremovalmodel_ExtraLayer/Bg_" + name + "_newadam" + \
        str(num_filter)+"cp-"+ lastit+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
            "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses + "_" + text_lr \
              + "_" + "val_loss"+ "_"+ text_susc +"_datagen" + "_evenlessbgnoartifacts_ExtraLayer_artif_1_nowrappingCircEager.ckpt"
"""
#path = "checkpoints/bgremovalmodel_ExtraLayer/Bg_BollmannExtralayer_newadam16cp-0005_trainsamples500_datasetiter5000" + \
#"_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayer_artif_1_nowrappingCircNew_2output.ckpt"

path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_Bgfrem_Bollmann_newadam16cp-0001_trainsamples500_datasetiter5000_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_RecCirc__datagenRecCirc.ckpt"


model_orig = tf.keras.models.load_model(path, compile=False)
model_orig.summary()


weights_orig = model_orig.get_weights()


###############################################################################
###############################################################################


#############################################
########## Create new model 
#############################################



#old - original seems to work
#input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_shape = (160, 160, 160, 1) # shape of pict, last is the channel

input_tensor = Input(shape = input_shape, name="input")

inputs = [Input(shape=input_shape), Input(shape=input_shape)]


LayerWBackgroundField = CreatebackgroundFieldLayer()

x = LayerWBackgroundField(inputs)

#outputs = build_CNN_BOLLMANinputoutput(x)

outputs = [x,build_CNN_BOLLMANinputoutput(x)]
model_new160 = Model(inputs, outputs)

model_new160.summary()


#model_new160.set_weights(weights_orig)
###############################################

#model_new.compile(optimizer=optimizerMINE, loss = lossU,run_eagerly=True)
#model.compile(optimizer=optimizerMINE, loss = lossU)


#model_new160.set_weights(weights_orig)

########################################
#############################################
########## Change for test############
########### only the final part  - background removal part
#############################################

 
#model_new160test = Model(inputs=model_new160.layers[2].output, outputs=[model_new160.outputs[0]])#1output
model_new160test = Model(inputs=model_new160.layers[2].output, outputs=[model_new160.outputs[1]])#2outputs

model_new160test.summary()
 
 
 ##
test_data = np.ones((1,) + input_shape)
test_1 = model_new160test.predict(test_data)


checkpoint = tf.train.Checkpoint(model_new160test)
status = checkpoint.restore(path) 
#status.expect_partial()



test_2 = model_new160test.predict(test_data) 
 
epsilon = 1e-5
print(np.all((np.abs(test_1-test_2) < epsilon))) 
 
 
 
model_new160test.save_weights('test.weights.h5')

 #########################################################################
 #########################################################################
 ########################################################################
 
 ##############################################################
###############################################################################
########## load nii.gz data
###############################################################################

import nibabel as nib

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
 



predictedPhase = model_new160test.predict(input_phasebgunwrap)
 
from plotting.visualize_volumes import visualize_all4, visualize_all4grey
from plotting.visualize_volumes import view_slices_3d, view_slices_3dNew

m_predicted, reference,error = visualize_all4(input_phasebgunwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0],
                                                  title = "dd", save=False, 
                                                  path="dd",
                                                   colormax=0.05,colormin=-0.05,
                                                   errormax = 0.1,errormin=-0.1  )


#####################################################################################


file_full = "datasynthetic/uniform02RectCircle_mask_phase160/testing/1samples.npz"

checkpoint = tf.train.Checkpoint(model_new160)
status = checkpoint.restore(path) 

loaded = np.load(file_full)
loaded =loaded['arr_0']
mask = loaded[0,:]
phase = loaded[1,:]
   #phasebg = loaded[2,:]
X_test = [mask[np.newaxis, :,:,:, np.newaxis], phase[np.newaxis, :,:,:, np.newaxis]]

y_pred = model_new160.predict(X_test)
   
pred_bgf = y_pred[0]
#pred_phase = y_pred[1]
pred_phase = y_pred[1]

   

   
   
import  matplotlib.pyplot as plt



predicted, reference,error = visualize_all4grey(pred_bgf[0,:,:,:,0], phase[:,:,:], pred_phase[0,:,:,:,0] ,
                                                   title = "dd" , save = False,
                                                   path = "dd",
                                                   colormax=0.2,colormin=-0.2,
                                                   errormax = 0.2,errormin=-0.2 )
   
   
   
predicted, reference,error = visualize_all4( pred_bgf[0,:,:,:,0], phase[:,:,:], pred_phase[0,:,:,:,0],
                                                   title = "dd" , save = False,
                                                   path = "ff",
                                                   colormax=0.2,colormin=-0.2,
                                                   errormax = 0.1,errormin=-0.1)
   


######################################################################
 
 ############################################################################
"""input_shape = (160, 160, 160, 1) # shape of pict, last is the channel

input_tensor = Input(shape = input_shape, name="input")

inputs = [Input(shape=input_shape), Input(shape=input_shape)]
 
 
########################################
#tinauer
input_shape = (160, 160, 160, 1)
input_tensor = tf.keras.Input(shape = input_shape, name='input')
output_tensor = build_CNN_BOLLMANinputoutput(input_tensor)

model = tf.keras.Model(input_tensor, output_tensor)

test_data = np.ones((1,) + input_shape)
test_1 = model.predict(test_data)"""


