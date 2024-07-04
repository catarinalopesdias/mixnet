#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:36:27 2024

loads a model and changes dimensions

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
from plotting.visualize_volumes import view_slices_3d, view_slices_3dNew

import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle
from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
###############################################################################
# Load existing model
###############################################################################



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
lastit="1000"

#checkpoint path


#Bg_BollmannExtralayer_newadam16cp-0052_trainsamples500_datasetiter2000_batchsize1_gaaccum10_loss_mse_01_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayer.ckpt

path = "checkpoints/bgremovalmodel_ExtraLayer/Bg_" + name + "_newadam" + \
        str(num_filter)+"cp-"+ lastit+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
            "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses + "_" + text_lr \
              + "_" + "val_loss"+ "_"+ text_susc +"_datagen" + "_evenlessbgnoartifacts_ExtraLayer_artif_1_nowrapping.ckpt"


model1 = tf.keras.models.load_model(path)

model1.summary()

############################################
##################################################






input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_shape = (160, 160, 160, 1) # shape of pict, last is the channel

input_tensor = Input(shape = input_shape, name="input")

#inputs = [Input(shape=input_shape)]




outputs = build_CNN_BOLLMANinputoutput(input_tensor)
print("newmodel")

modelNEW = Model(input_tensor, outputs)

pathmodel  = "models/backgroundremovalBOLLMAN_ExtraLayer/model_BR_BollmannExtralayer_newadam_16filters_trainsamples500_datasetiter2000_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayerartif.keras"
print("loadweights keras")

modelNEW.load_weights(pathmodel).expect_partial()
#LayerWBakgroundField = CreatebackgroundFieldLayer()

#x = LayerWBakgroundField(inputs)



