"""#Created on Mon Nov 20 10:03:11 2023

@author: catarinalopesdias
"""

# load and evaluate a saved model
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
#import matplotlib.pyplot as plt
#import numpy as np
from datahandling import read_and_decode_tf
from plotting.visualize_volumes import visualize_all4
from plotting.visualize_volumes import visualize_all
from tensorflow import keras
import numpy as np
###############################################################################
# Load existing model
###############################################################################
# parameter models
num_train_instances = 500
dataset_iterations = 5000
#finalcheckpoint = 2996
batch_size = 1
gaaccumsteps = 10
num_filter = 16
losses = ["mse", "mse"]
text_stop = "stopping"
lr =0.003
text_lr = str(lr).split(".")[1]


path = "checkpoints/doublenet/_BollmannPhillip_newadam" + \
        str(num_filter)+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + \
            "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + losses[0] +losses[1]+ \
                text_stop+"_"+text_lr+"_valloss.ckpt"


model = tf.keras.models.load_model(path)


model.compile(loss = losses[0], optimizer = 'adam')



#################################################################
#################################################################

##############################################################
### Preprocess data for testing
#################################################################
tfrecord_files_tst = []
#folder_test = "tst_synthetic50"
folder_test = "tst_2019"

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

counter = 0
for data_sample_tst in dataset_test_np:
    input_ds_tst, output_ds_tst = data_sample_tst
    

    input_file = input_ds_tst[np.newaxis, :,:,:, np.newaxis]
    
    predicted = model.predict(input_file)

    counter+=1
    print(f"Test set {counter}")




    
    #prediction_title = "bg removal" +  "epoch" + str(counter)
    #print(prediction_title)
   #m_predicted, reference,error = visualize_all4(input_file[0,:,:,:,0], output_ds_tst[:,:,:], predicted[0][0,:,:,:,0]   ,title = prediction_title, save="False", path="dd" )
    prediction_title ="fullmodel" + "epoch" + str(counter)
    m_predicted, reference,error = visualize_all4(input_file[0,:,:,:,0], output_ds_tst[:,:,:], predicted[1][0,:,:,:,0]   ,title = prediction_title, save="False", path="dd" )
