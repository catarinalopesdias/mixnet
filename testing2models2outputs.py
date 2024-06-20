#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:38:10 2024

@author: catarinalopesdias
"""

num_train_instances = 500

# validation set 
samples_dic = []

for i in range(num_train_instances):
  samples_dic.append( str(i) + 'samples' )
   

#######################
# create partition
#########################
partition_factor = 0.8

partition = {'train': samples_dic[0: int(partition_factor * num_train_instances)] , 
             'validation': 
            samples_dic[ -int(( 1-partition_factor) * num_train_instances): num_train_instances]}
#################################################
    

###################################
# create data generator#############
###################
from  my_classes.dataGenerator.DataGenerator_susc02_bgrem_evenlessbgnoartifacts_onlyphaseExtraLayer import DataGeneratorUniformevenlessbgnoartifactsExtraLayer

# Generators
#text regarding susc
text_susc="unif02"
#training set 
training_generatorUnif   = DataGeneratorUniformevenlessbgnoartifactsExtraLayer(partition['train'])
#validation set
validation_generatorUnif = DataGeneratorUniformevenlessbgnoartifactsExtraLayer(partition['validation'])


##################################################################
# model 1 and model 2
######################################################

from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
from my_classes.keraslayer.layerbackgroundfield import CreatebackgroundFieldLayer
import os
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle


input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")

inputs = [Input(shape=input_shape), Input(shape=input_shape)]


LayerWBakgroundField = CreatebackgroundFieldLayer()

x = LayerWBakgroundField(inputs)


#LayerWBakgroundField = CreatebackgroundFieldLayer()
#output = LayerWBakgroundField(inputs)
#x= output

#outputs = build_CNN_BOLLMANinputoutput(x)
outputs1 = build_CNN_BOLLMANinputoutput(x)
#model1 = Model(inputs, outputs1)
name = "BollmannExtralayer"


outputs2 = [x, build_CNN_BOLLMANinputoutput(x)]

model1 = Model(inputs,outputs1)
model2 = Model(inputs,outputs2)


Bg_BollmannExtralayer_newadam16cp-0563_trainsamples500_datasetiter2000_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayer.ckpt