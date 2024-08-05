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
from plotting.visualize_volumes import view_slices_3d, view_slices_3dNew

import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle

###############################################################################
# Load existing model
###############################################################################


# model data


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
"""
#checkpoint path


path = "checkpoints/preprocessing_bgremovalmodel/Bg_PhaseBgf_Bgfrem_Bollmann_newadam16cp-0001_trainsamples500_datasetiter5000_batchsize1_gaaccum10_loss_mse_001_val_loss_unif02_RecCirc__datagenRecCirc.ckpt"


model1 = tf.keras.models.load_model(path)
model1.summary()

#model.compile(loss = losses, optimizer = 'adam')

########################
# create new model
###################################
#from keras.layers import Input
from keras.models import Model
from keras.layers import Input

#model2Synthetic = Model(inputs=model1.inputs, outputs=[model1.layers[2].output, model1.outputs[0]])
#model2Synthetic.layers[2].size = 160
#model2Synthetic.compile(loss = losses, optimizer = 'adam')


########################
# create new model
###################################
#from keras.layers import Input
#from keras.models import Model
#model2 = Model(inputs=model.inputs, outputs=[model.layers[2].output, model.outputs[0]])






#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################
newdata=True
import  matplotlib.pyplot as plt



path_common_init = "models/preprocessing_backgroundremoval/prediction_images/unif"

network_type = "PhaseBgf_Bgfrem_Bollmann_newadam16"

for epoch_i in range(3): #num_instance
   file =str(epoch_i)+"samples"

   if newdata:
        file =str(epoch_i)+"samples"
        #loaded = np.load(fullfile)
        text_typedata = "testdata"
        file_full = "datasynthetic/uniform02RectCircle-Mask-MaskedPhase-Phase/testing/" + file + ".npz"

   else: #traindata
        text_typedata = "traindata" 
        file_full = "datasynthetic//uniform02RectCircle-Mask-MaskedPhase-Phase/training/" + file + ".npz"

   loaded = np.load(file_full)
   loaded =loaded['arr_0']
   #mask = loaded[0,:]
       #loaded[1,:] masked phase
   phase = loaded[2,:] #phase 
   #phasebg = loaded[2,:]
   X_test =  phase[np.newaxis, :,:,:, np.newaxis]

   y_pred = model1.predict(X_test)
   
   #mask = y_pred[0]
   pred_phasemask = y_pred[0]

   pred_bgf = y_pred[1]
   pred_phase = y_pred[2]

   #plt.imshow(X_test[0,64,:,:,0], cmap='gray',  vmin=-0.4, vmax=0.4)   
   #plt.imshow(y_pred[0,64,:,:,0], cmap='gray',  vmin=-0.01, vmax=0.01)   

   #plt.imshow(phase[64,:,:], cmap='gray',  vmin=-0.4, vmax=0.4)   
   plt.imshow(X_test[0, 64,:,:], cmap='gray',  vmin=-0.4, vmax=0.4)   
   plt.show()
   #plt.imshow(mask[0,64,:,:,0], cmap='gray',  vmin=-0.4, vmax=0.4)   
   #plt.show()
   plt.imshow(pred_phasemask[0,64,:,:,0], cmap='gray',  vmin=-0.2, vmax=0.2)   
   plt.show()
   plt.imshow(pred_bgf[0,64,:,:,0 ], cmap='gray',  vmin=-0.4, vmax=0.4)   
   plt.show()
   plt.imshow(pred_phase[0,64,:,:,0 ], cmap='gray',  vmin=-0.4, vmax=0.4)   
   plt.show()


"""


   path_common_final =  str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
                  "_batchsize"+ str(batch_size) + "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses +"_"+text_lr +\
                      "_"  +  "valloss"+"_datagen_"+ text_typedata +"_" + lastit+ "_epoch" + str(epoch_i) + "_evenlessbgttt_nowrappingCirC"
                  
                  
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
   
   








"""





"""model2Real = Model(inputs=model1.layers[2].output, outputs=[model1.outputs[0]])

#model2Real = Model(inputs=[model1.layers[2].output], outputs=[model1.outputs[0]])
model2Real.summary()
#model2Real.layers.pop(0)
#model1.summary()
#model1.layers.pop(0)
#model1.layers.pop(0)
#model1.summary()
#model2Real.layers.pop(0)
model2Real.summary()

model_config = model2Real.get_config()
model_config["layers"][0]["config"]["batch_input_shape"] = (160, 160,160,1)
modified_model = tf.keras.Model.from_config(model_config)
modified_model.summary()
input_tensor = Input(shape = (160, 160, 160, 1), name="input_real")
#model1(input_tensor)
#model1.summary()

newOutputs = model2Real(input_tensor)
newModel = Model(input_tensor, newOutputs)


#model2Real = Model(inputs=[input_tensor], outputs=[model2Real.outputs[0]])
#model2Real.summary()
                    
#input_tensor = Input(shape = (160, 160, 160, 1), name="input_real")
#print(model1.layers[3].inputs)
#model1.layers[3](input_tensor)
#print(model1.layers[3].inputs)
#model2Real = Model(inputs=input_tensor, outputs=[model1.outputs[0]])
#model2Real.summary()

#model2Real = Model(inputs=[model1.layers[2].output,     , outputs=[model1.outputs[0]])

#################################################################
#################################################################

# load unseen data 
################################################
#   Import data
################################################

from datahandling import read_and_decode_tf
import nibabel as nib


path_common_init = "models/backgroundremovalBOLLMAN_ExtraLayer/prediction_images/unif/"
network_type = "BgRem_BollmannExtralayer_newadam"
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

#bgf, predictedPhase = model2Synthetic.predict(mask,reference_phasetissue)
#create crop
#reference_phasetissue = reference_phasetissue[:,0:128,0:128,0:128,:]
#input_phasebgwrap = input_phasebgwrap[:,0:128,0:128,0:128,:]
#input_phasebgunwrap = input_phasebgunwrap[:,0:128,0:128,0:128,:]
###################################################################################




predictedPhase = model2Real.predict(input_phasebgunwrap)

prediction_title = "nii.gz files  - qsm2016"
pathi = path_common_init +"color/real/"+ network_type  + "niigz_nowrapping"

#m_predicted, reference,error = visualize_all4(input_phasebgwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0],
#                                                  title = prediction_title, save=True, 
#                                                  path=pathi,
#                                                   colormax=0.2,colormin=-0.2,
#                                                   errormax = 0.1,errormin=-0.1  )


m_predicted, reference,error = visualize_all4(input_phasebgunwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0],
                                                  title = prediction_title, save=True, 
                                                  path=pathi,
                                                   colormax=0.05,colormin=-0.05,
                                                   errormax = 0.1,errormin=-0.1  )

pathi = path_common_init +"grey/real/"+ network_type  + "niigz_nowrapping"

predicted, reference,error = visualize_all4grey(input_phasebgunwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0] ,
                                                   title = "ff" , save = False,
                                                   path = pathi,
                                                   colormax=0.05,colormin=-0.05,
                                                   errormax = 0.13,errormin=-0.13, slice_nr=64 )


predicted, reference,error = visualize_all4grey(input_phasebgunwrap[0,:,:,:,0], reference_phasetissue[0,:,:,:,0], predictedPhase[0,:,:,:,0] ,
                                                   title = "ff" , save = True,
                                                   path = pathi,
                                                   colormax=0.05,colormin=-0.05,
                                                   errormax = 0.13,errormin=-0.13, slice_nr=100 )
   

maskCropped = mask[0:128,0:128,0:128]
###########################################################################
#####################################
# apllying a black mask 
input_phasebgunwrapCropped= input_phasebgunwrap[0,0:128,0:128,0:128,0]
view_slices_3d(input_phasebgunwrapCropped, 100, -0.05, 0.05, title='ff')



input_phasebgunwrapCroppedMask = input_phasebgunwrapCropped*maskCropped

input_phasebgunwrapCroppedMask[input_phasebgunwrapCroppedMask == 0] =-100




view_slices_3d(input_phasebgunwrapCroppedMask, 100, -0.05, 0.05, title='ff')
   
   
###############################
reference_phasetissueCropped = reference_phasetissue[0,0:128,0:128,0:128,0]
view_slices_3d(reference_phasetissueCropped, 100, -0.05, 0.05, title='ff')
reference_phasetissueCroppedMask = reference_phasetissueCropped*maskCropped
reference_phasetissueCroppedMask[reference_phasetissueCroppedMask == 0] =-100
view_slices_3d(reference_phasetissueCroppedMask, 100, -0.05, 0.05, title='ff')
###########################
predictedPhaseCropped = predictedPhase[0,0:128,0:128,0:128,0]
view_slices_3d(predictedPhaseCropped, 100, -0.05, 0.05, title='ff')
predictedPhaseCroppedMask = predictedPhaseCropped*maskCropped
predictedPhaseCroppedMask[predictedPhaseCroppedMask == 0] =-100
view_slices_3d(predictedPhaseCroppedMask, 100, -0.05, 0.05, title='ff')
"""

"""
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
"""