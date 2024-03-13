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

from networks.network_phillip_inputoutput import build_CNN_phillip_inputoutput
from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
from keras.models import Model
from keras.initializers import GlorotNormal
from keras.callbacks import ModelCheckpoint
import pickle
from newGA import GradientAccumulateModel
#from network_NEW import build_CNN_NEW
from create_datasetfunctions import simulate_susceptibility_sources, generate_3d_dipole_kernel, forward_convolution
from plotting.visualize_volumes import view_slices_3dNew
from  my_classes.DataGenerator import DataGenerator

#############################################################
##         Import data#################

#loaded = np.load('datasynthetic/115samples.npz')
#loaded.files
#phase = loaded['phase1']
#phase_bg = loaded['phase_bg1']
#gt = loaded['sim_gt1']
#del loaded
########################################

#num_train_instances = phase.shape[0]
num_train_instances = 4
##############################
loss_model1 = "mse"    #mse "mean_absolute_error"
loss_model2 = "mse"    #mse "mean_absolute_error"


###############################

params = {'dim': (128,128,128),
          'batch_size': 1,
          'n_channels': 1,
          'shuffle': True}

###################################
#create dictonary
###################################
#phasebg_dic =[]
#gt_dic = []
samples_dic = []


for i in range(num_train_instances):
    #phasebg_dic.append( str(i) + 'samples' + '.npy')
    #gt_dic.append('gt-' + str(i)+'.npy')
    samples_dic.append( str(i) + 'samples' )
   

#######################
partition_factor = 0.5
#partition = {'train': phasebg_dic[0: int(partition_factor * num_train_instances)] , 
#             'validation':  phasebg_dic[int(-partition_factor * num_train_instances): num_train_instances]}

partition = {'train': samples_dic[0: int(partition_factor * num_train_instances)] , 
             'validation':  samples_dic[int(-partition_factor * num_train_instances): num_train_instances]}

#labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
#########################################################


##################

#labels = tools.read_dict('../datasets/dataset_single.csv',
#                                 value_type=constants.ValueType.INT)




# Generators
training_generator   = DataGenerator(partition['train'])
validation_generator = DataGenerator(partition['validation'])
######################################################################################
######################################################################################


#################################

#learningrate
lr =0.003
text_lr = str(lr).split(".")[1]
#################################
# optimizer
optimizerMINE = Adam(
              learning_rate=lr,
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



#using module class
#model1 = build_CNN_BOLLMAN(input_tensor)
#model2 = build_CNN_BOLLMAN(model1.output)
#full_model  = Model(inputs=new_model.input, outputs=output)
#model = Model(input_tensor, model2) 

#using no ~Module class~
ushape1 = build_CNN_BOLLMANinputoutput(input_tensor)
ushape2 = build_CNN_phillip_inputoutput(tf.keras.backend.stop_gradient(ushape1))  #ushape1 ---- tf.stop_gradient(ushape1) tf.keras.backend.stop_gradient
#model = Model(input_tensor, [ushape1,ushape2]) #maybe ushape2.input -- [ushape1,ushape2] ----  [ushape1]
model = Model(input_tensor, outputs={'ushape1': ushape1, 'ushape2': ushape2}) #maybe ushape2.input -- [ushape1,ushape2] ----  [ushape1]





text_stop = "stopping"

losses = [loss_model1, loss_model2]

print("Model with gradient accumulation")
gaaccumsteps = 10;
model = GradientAccumulateModel(accum_steps=gaaccumsteps, inputs=model.input, outputs=model.outputs )
#odel = GradientAccumulateModel(accum_steps=gaaccumsteps, inputs=model.input, outputs={'ushape1': model.outputs[0], 'ushape2': model.outputs[1]} )


model.summary
model.compile(optimizer=optimizerMINE, loss = losses)
##################

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
    
    del gpus
###############################################################################

dataset_iterations = 5000
save_period = 100
batch_size = 1
num_filter = 16

###############
 # "cp-{epoch:04d}"+
checkpoint_path = "checkpoints/doublenet/_BollmannPhillip_newadam" + str(num_filter)+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses[0] +losses[1]+text_stop+"_"+text_lr+".ckpt"

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

"""
train_images_m1 =tf.expand_dims(phase_bg, 4)
train_labels_m1 = tf.expand_dims(phase, 4)

#train_images_m2 =
#train_images_m2 = tf.expand_dims(phase, 4)
train_labels_m2 = tf.expand_dims(gt, 4)


print("fit model")

#
train_labels= [train_labels_m1, train_labels_m2] #[train_labels_m1]#
del train_labels_m1, train_labels_m2 #train_images_m2
"""

##########################################







history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)


#history = model.fit(train_images_m1,train_labels,  epochs=dataset_iterations, batch_size=batch_size, shuffle=True,
#          callbacks = [cp_callback,earlystop],
#          #validation_split=0.1
#          )  # pass callback to training for saving the model80

loss_historyGA = history.history['loss']


with open('loss_historyGA.pickle', 'wb') as f:
    pickle.dump([loss_historyGA, dataset_iterations], f)
"""
###################

#save model
if not os.path.exists("models/doublenet"): 
    os.makedirs("models/doublenet") 

model_name = "models/doublenet/model_BollmannPhillip_newadamBOTH_" + str(num_filter)+"filter_trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + "_loss_"+ losses[0] +losses[1] +"_bollmannphillipp_"+text_stop+"_"+text_lr+ ".keras"

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
lossnamefile = "models/doublenet/loss/model_BollmannPhillip_newadamBOTH_" + str(num_filter)+"filter_trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1] + "_bollmannphillip_"+text_stop+"_"+text_lr


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

loss_historyF = history.history['conv3d_transpose_129_loss']

plt.figure(figsize=(6, 3))
plt.plot(loss_historyF)
#plt.ylim([0, 0.03])
plt.title("Loss first model")
plt.xlabel("Epochs ")
lossnamefileF = "models/doublenet/loss/model_BollmannPhillip_newadam_1_" + str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1]+text_stop+"_"+text_lr

plt.savefig(lossnamefileF + lossfile_extensionpng )

lossfile_extension =".txt"

file = open(lossnamefileF + lossfile_extension,'w')
for item in loss_historyF:

	file.write(str(item)+"\n")
file.close()

# Second loss ######################################

loss_historyS = history.history['add_63_loss']

plt.figure(figsize=(6, 3))
plt.plot(loss_historyS)
#plt.ylim([0, 0.03])
plt.title("Loss Second model")
plt.xlabel("Epochs ")
lossnamefileS = "models/doublenet/loss/model_BollmannPhillip_newadam_2_" + str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ losses[0] +losses[1]+text_stop+"_"+text_lr

plt.savefig(lossnamefileS + lossfile_extensionpng )

lossfile_extension =".txt"

file = open(lossnamefileS + lossfile_extensiontxt,'w')
for item in loss_historyS:

	file.write(str(item)+"\n")
file.close()

"""

