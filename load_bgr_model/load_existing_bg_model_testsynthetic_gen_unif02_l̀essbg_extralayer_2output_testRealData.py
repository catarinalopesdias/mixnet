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
lastit="2000"

#checkpoint path


#Bg_BollmannExtralayer_newadam16cp-0052_trainsamples500_datasetiter2000_batchsize1_gaaccum10_loss_mse_01_val_loss_unif02_datagen_evenlessbgnoartifacts_ExtraLayer.ckpt

path = "checkpoints/bgremovalmodel_ExtraLayer/Bg_" + name + "_newadam" + \
        str(num_filter)+"cp-"+ lastit+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
            "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses + "_" + text_lr \
              + "_" + "val_loss"+ "_"+ text_susc +"_datagen" + "_evenlessbgnoartifacts_ExtraLayer_artif.ckpt"



model1 = tf.keras.models.load_model(path)
#model = tf.keras.saving.load_model(path)


#model.compile(loss = losses, optimizer = 'adam')

########################
# create new model
###################################
#from keras.layers import Input
from keras.models import Model
model2Synthetic = Model(inputs=model1.inputs, outputs=[model1.layers[2].output, model1.outputs[0]])
#model2Synthetic.layers[2].size = 160
#model2Synthetic.compile(loss = losses, optimizer = 'adam')


model2Real = Model(inputs=model1.layers[2].output, outputs=[model1.outputs[0]])



#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################

from datahandling import read_and_decode_tf

##############################################################
### Preprocess data for testing - tst_2019
#################################################################
tfrecord_files_tst = []
#folder_test = "tst_synthetic50"
############################################################
folder_test = "tst_2019"
##############################################################

tfrecord_dir_tst = "datareal/" + folder_test

# List all files in the directory
files_in_tst_directory = os.listdir(tfrecord_dir_tst)

for file in files_in_tst_directory:
    if file.endswith(".tfrecords"):
        full_path =os.path.join(tfrecord_dir_tst, file)
        tfrecord_files_tst.append(full_path)

tfrecord_dataset_tst = tf.data.TFRecordDataset(tfrecord_files_tst)

# Apply the parsing function to each record
parsed_dataset_tst = tfrecord_dataset_tst.map(read_and_decode_tf)

##################################################################
############ Predict and plot original data
#################################################################
dataset_test_np = parsed_dataset_tst.as_numpy_iterator()

path_common_init = "models/backgroundremovalBOLLMAN_ExtraLayer/prediction_images/real/"

network_type = "BgRem_BollmannExtralayer_newadam"

counter = 0
for data_sample_tst in dataset_test_np:
    input_ds_tst, output_ds_tst = data_sample_tst
    

    input_file = input_ds_tst[np.newaxis, :,:,:, np.newaxis]
    
    predicted = model2Real.predict(input_file)

    #pred_bgf = predicted[0]
    pred_phase = predicted#[0]
   
    counter+=1
    print(f"Test set {counter}")



    path_common_final =  str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
                   "_batchsize"+ str(batch_size) + "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses +"_"+text_lr +\
                       "_"  +  "valloss"+"_datagen_"+  "_epoch" + str(counter) +"_" + lastit+ "_evenlessbg" 
   
    pathi = path_common_init +"/color/"+ network_type  + path_common_final+"_color"
    
    prediction_title ="tst-2019 " + "- epoch " + str(counter)
    m_predicted, reference,error = visualize_all4(input_file[0,:,:,:,0], output_ds_tst[:,:,:], pred_phase[0,:,:,:,0],
                                                  title = prediction_title, save=True, 
                                                  path=pathi,
                                                   colormax=0.2,colormin=-0.2,
                                                   errormax = 0.2,errormin=-0.2
                                                  )


###########################################################
###############################################
#bla = tf.data.TFRecordDataset("datareal/tst_2019/1.tfrecords")

#print("bla")
#read_and_decode_tf(bla)

#parsed = bla.map(read_and_decode_tf)
#datanp = parsed.as_numpy_iterator()

#for i in datanp:
#    input1, output1  = i    
###########################################

###############################################################################
########## load nii.gz data
###############################################################################

folder_dir ="datareal/qsm2016_recon_challenge/data/"
file = "msk.nii.gz"
filetoupload = folder_dir + file
import nibabel as nib
img = nib.load(filetoupload)

mask = img.get_fdata()
mask = mask[np.newaxis, :,:,:, np.newaxis]

###############
file = "phs_tissue.nii.gz"
filetoupload = folder_dir + file
img = nib.load(filetoupload)
reference_phasetissue = img.get_fdata()
reference_phasetissue = reference_phasetissue[np.newaxis, :,:,:, np.newaxis]


file = "phs_wrap.nii.gz"
filetoupload = folder_dir + file
img = nib.load(filetoupload)
input_phasebgwrap = img.get_fdata()
input_phasebgwrap = input_phasebgwrap[np.newaxis, :,:,:, np.newaxis]



bgf, predictedPhase = model2Synthetic.predict(mask,reference_phasetissue)

predictedPhase = model2Real.predict(input_phasebgwrap)

prediction_title = "nii.gz files  - qsm2016"
pathi = path_common_init +"/color/"+ network_type  + "niigz"

m_predicted, reference,error = visualize_all4(input_phasebgwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0],
                                                  title = prediction_title, save=True, 
                                                  path=pathi,
                                                   colormax=0.2,colormin=-0.2,
                                                   errormax = 0.1,errormin=-0.1  )




   
   
   
   