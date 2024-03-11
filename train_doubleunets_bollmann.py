#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# new preprocessing + training +gradient accumulation
"""
Created on Wed Dec  6 09:28:17 2023

@author: catarinalopesdias
"""



import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import matplotlib
from matplotlib import transforms
from scipy import ndimage
import os
import tensorflow as tf

from keras.optimizers import Adam
from keras.layers import Input

from networks.network_adaptedfrom_BOLLMAN import build_CNN_BOLLMAN
from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
from keras.models import Model
from keras.initializers import GlorotNormal
from keras.callbacks import ModelCheckpoint
import pickle
from newGA import GradientAccumulateModel
#from network_NEW import build_CNN_NEW
from create_datasetfunctions import simulate_susceptibility_sources, generate_3d_dipole_kernel, forward_convolution
from plotting.visualize_volumes import view_slices_3dNew

#############################################################
##         Import data#################

loaded = np.load('datasynthetic/150samples.npz')
#loaded.files
phase = loaded['phase1']
phase_bg = loaded['phase_bg1']
gt = loaded['sim_gt1']
del loaded
########################################

num_train_instances = phase.shape[0]
##############################
loss_model1 = "mse"#mse "mean_absolute_error"
loss_model2 = "mse"#mse "mean_absolute_error"

print("Model with gradient accumulation")
gaaccumsteps = 10;
###############################
optimizerMINE = Adam(
              learning_rate=0.0003,
                beta_1=0.9,
                beta_2=0.999,
               epsilon=1e-8
               ) 
###############################################
# Model 1: Compile  model 
###############################################

print("compile model")
input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")



#from networks.network_phillip import build_CNN

#using model class
#model1 = build_CNN_BOLLMAN(input_tensor)
#model2 = build_CNN_BOLLMAN(model1.output)
#full_model  = Model(inputs=new_model.input, outputs=output)
#model = Model(input_tensor, model2) 

#using no ~Module class~
ushape1 = build_CNN_BOLLMANinputoutput(input_tensor)
ushape2 = build_CNN_BOLLMANinputoutput((ushape1))  #ushape1 ---- tf.stop_gradient(ushape1) tf.keras.backend.stop_gradient
model = Model(input_tensor, [ushape1,ushape2]) #maybe ushape2.input -- [ushape1,ushape2] ----  [ushape1]

losses = [loss_model1, loss_model2]

model = GradientAccumulateModel(accum_steps=gaaccumsteps, inputs=model.input, outputs=model.outputs )


#model = Model(input_tensor, ushape2.outputs) #get from orig deepQSM algorithm


model.summary


model.compile(optimizer=optimizerMINE, loss = losses)
##################



###############################################################################
#   Visualize outcomes of untrained models
#  Untrained models 
############################################################################### 



###############################################################################
###############################################################################
#   training part
###############################################################################

#   test GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
else:
    print("No GPU devices found.")
###############################################################################

dataset_iterations = 5000
save_period = 100
batch_size = 1
num_filter = 16

###############
 # "cp-{epoch:04d}"+
checkpoint_path = "checkpoints/doublenet/newadam" + str(num_filter)+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses[0] +losses[1]+"nostopping.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only = False,
                                                 #save_freq=save_period,
                                                 save_freq="epoch",
                                                 save_best_only=True,
                                                 monitor = "loss",
                                                 verbose=1)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",#
    min_delta=0,
    patience=100,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

##############################
##############################
##############################


train_images_m1 =tf.expand_dims(phase_bg, 4)
train_labels_m1 = tf.expand_dims(phase, 4)

#train_images_m2 =
train_images_m2 = tf.expand_dims(phase, 4)
train_labels_m2 = tf.expand_dims(gt, 4)


print("fit model")

#
train_labels= [train_labels_m1, train_labels_m2] #[train_labels_m1]#[
history = model.fit(train_images_m1,train_labels,  epochs=dataset_iterations, batch_size=batch_size, shuffle=True,
          callbacks = [cp_callback,earlystop])  # pass callback to training for saving the model80

loss_historyGA = history.history['loss']


with open('loss_historyGA.pickle', 'wb') as f:
    pickle.dump([loss_historyGA, dataset_iterations], f)

###################

#save model
if not os.path.exists("models/doublenet"): 
    os.makedirs("models/doublenet") 

model_name = "models/doublenet/model_newadamBOTH_" + str(num_filter)+"filter_trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + "_loss_"+ losses[0] +losses[1] +"nostopping.keras"

model.save(model_name)


#tf.keras.models.save_model(model, model_name)    
    
if not os.path.exists("models/doublenet/loss"): 
    os.makedirs("models/doublenet/loss") 
        
###################
# plot loss figure
plt.figure(figsize=(6, 3))
plt.plot(loss_historyGA)
#plt.ylim([0, 0.03])
plt.title("Loss")
plt.xlabel("Epochs ")
lossnamefile = "models/doublenet/loss/model_newadamBOTH_" + str(num_filter)+"filter_trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1] + "nostopping"


lossfile_extensiontxt =".txt"
lossfile_extensionpng =".png"


plt.savefig(lossnamefile + lossfile_extensionpng )



###############
# save loss as txt

file = open(lossnamefile + lossfile_extensiontxt,'w')
for item in loss_historyGA:

	file.write(str(item)+"\n")
file.close()
#######################################
# First loss

loss_historyF = history.history['conv3d_transpose_35_loss']

plt.figure(figsize=(6, 3))
plt.plot(loss_historyF)
#plt.ylim([0, 0.03])
plt.title("Loss first model")
plt.xlabel("Epochs ")
lossnamefileF = "models/doublenet/loss/model_newadam_1_" + str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1]+"nostop"

plt.savefig(lossnamefileF + lossfile_extensionpng )

lossfile_extension =".txt"

file = open(lossnamefileF + lossfile_extension,'w')
for item in loss_historyF:

	file.write(str(item)+"\n")
file.close()

# Second loss ######################################

loss_historyS = history.history['conv3d_transpose_40_loss']

plt.figure(figsize=(6, 3))
plt.plot(loss_historyS)
#plt.ylim([0, 0.03])
plt.title("Loss Second model")
plt.xlabel("Epochs ")
lossnamefileS = "models/doublenet/loss/model_newadam_2_" + str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1]+"nostop"

plt.savefig(lossnamefileS + lossfile_extensionpng )

lossfile_extension =".txt"

file = open(lossnamefileS + lossfile_extensiontxt,'w')
for item in loss_historyS:

	file.write(str(item)+"\n")
file.close()



