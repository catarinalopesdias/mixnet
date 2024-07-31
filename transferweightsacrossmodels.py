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

#############################################
########## Load checkpoint
#############################################


# model data
num_train_instances = 500
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


model_orig = tf.keras.models.load_model(path)
model_orig.summary()

#model.compile(loss = losses, optimizer = 'adam')

weights_orig = model_orig.get_weights()


###############################################################################
###############################################################################


#############################################
########## Create new model 
#############################################



#old - original seems to work
#input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_shape = (128, 128, 128, 1) # shape of pict, last is the channel

input_tensor = Input(shape = input_shape, name="input")

inputs = [Input(shape=input_shape), Input(shape=input_shape)]


LayerWBakgroundField = CreatebackgroundFieldLayer()

x = LayerWBakgroundField(inputs)


#LayerWBakgroundField = CreatebackgroundFieldLayer()
#output = LayerWBakgroundField(inputs)
#x= output

#outputs = build_CNN_BOLLMANinputoutput(x)
outputs = build_CNN_BOLLMANinputoutput(x)
model_new160 = Model(inputs, outputs)

model_new160.summary()


#model_new160.set_weights(weights_orig)
###############################################
###############################################
print("Model with gradient accumulation")
gaaccumsteps = 10;
#learningrate
lr =0.001 #0.0004
text_lr = str(lr).split(".")[1]

model_new160 = GradientAccumulateModel(accum_steps=gaaccumsteps,
                                inputs=model_new160.input, outputs=model_new160.output)

lossU = "mse" #"mean_absolute_error"#"mse"# "mean_absolute_error" #"mse"    #mse

#optimizer
optimizerMINE = Adam(
              learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
               epsilon=1e-8
               ) 



#model_new.compile(optimizer=optimizerMINE, loss = lossU,run_eagerly=True)
#model.compile(optimizer=optimizerMINE, loss = lossU)

model_new160.summary()

model_new160.set_weights(weights_orig)

########################################
#############################################
########## Change for test
#############################################

 
model_new160test = Model(inputs=model_new160.layers[2].output, outputs=[model_new160.outputs[0]])

model_new160test.summary()
 
 
 
model_new160test.set_weights(weights_orig)
 
 ############################################################################
input_shape = (160, 160, 160, 1) # shape of pict, last is the channel

input_tensor = Input(shape = input_shape, name="input")

inputs = [Input(shape=input_shape), Input(shape=input_shape)]
 
 
fg = tf.keras.models.clone_model(
    model_orig,
    input_tensors = inputs,
    clone_function=lambda x: x)

fg.summary()
