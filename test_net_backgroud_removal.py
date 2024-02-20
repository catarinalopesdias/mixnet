#
"""
Created on Mon Feb 19 09:18:00 2024

@author: catarinalopesdias
Creates synthetic data
trains 1 Unet network 
compares results
"""
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import matplotlib
from matplotlib import transforms
from scipy import ndimage
import os
import tensorflow as tf


from keras.layers import Input ,Conv3D, Conv3DTranspose, LeakyReLU, UpSampling3D, Concatenate , Add, BatchNormalization
from keras.models import Model
from keras.initializers import GlorotNormal
from keras.callbacks import ModelCheckpoint
import pickle

from newGA import GradientAccumulateModel
from network import build_CNN
from create_datasetfunctions import simulate_susceptibility_sources, generate_3d_dipole_kernel, forward_convolution
from visualize_volumes import view_slices_3dNew

################################################
#   Import data
################################################
#view input
#view output

phase_bg = np.load('syntheticdata/phase_bg100.npy')
phase = np.load('syntheticdata/phase100.npy')

view_slices_3dNew(phase[3,:,:,:], 50, 50,50, vmin=-10, vmax=10, title="phase") 
view_slices_3dNew(phase_bg[3,:,:,:], 50, 50,50, vmin=-10, vmax=10, title="phase+bg")

num_train_instances = phase.shape[0]

##############################


#

#############################################################
#### Compile the model
#############################################################
input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")

#ushape1 = build_CNN(input_tensor)
#ushape2 = build_CNN(ushape1)

print("compile model")

#model = Model(input_tensor, ushape2) #get from orig deepQSM algorithm
model = build_CNN(input_tensor)
#model.compile(optimizer='adam', loss='mean_absolute_error')
#model.summary()

###############################################
###############################################
print("Model with gradient accumulation")
gaaccumsteps = 8;
model = GradientAccumulateModel(accum_steps=gaaccumsteps, inputs=model.input, outputs=model.output)

# we changed to mean absolute error
lossU = "mean_absolute_error"#"mse"# "mean_absolute_error" #"mse"    #mse
model.compile(optimizer='adam', loss= lossU) #mean_absolute_error
model.summary()

#################################################################################

###############################################################################
###############################################################################
print("untrained")
# what does the untrained model predict
test_epoch_nbr = 3
X_test = phase_bg[np.newaxis, test_epoch_nbr,:,:,:, np.newaxis]
print(X_test.shape)

y_pred = model.predict(X_test)

print(y_pred.shape)


view_slices_3dNew(phase_bg[test_epoch_nbr, :, :, :], 50,50,50, vmin=-1, vmax=1, title=' Phase and bg')
view_slices_3dNew(phase[test_epoch_nbr, :, :, :],50,50,50 , vmin=-1, vmax=1, title='phase')
view_slices_3dNew(y_pred[0, :, :, :, 0], 50,50,50, vmin=-1, vmax=1, title='Predicted phase')


#view_slices_3d(sim_fw_full[test_epoch_nbr, :, :, :], slice_nbr=16, vmin=-1, vmax=1, title='Input Tissue Phase')
#view_slices_3d(sim_gt_full[test_epoch_nbr, :, :, :], slice_nbr=16, vmin=-1, vmax=1, title='GT Susceptibility')
#view_slices_3d(y_pred[0, :, :, :, 0], slice_nbr=16, vmin=-1, vmax=1, title='Predicted Susceptibility')

################################################################################
# training part
##########################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)
else:
    print("No GPU devices found.")


##############################################################################################


dataset_iterations = 100
save_period = 50
batch_size = 2

###############


# train

checkpoint_path = "checkpoints/bgremovalmodel/cp-{epoch:04d}"+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+".ckpt"



checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=False,
                                                 save_freq=save_period,
                                                 monitor = "val_loss",
                                                 verbose=1)

earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)




train_images =tf.expand_dims(phase_bg, 4)
train_labels = tf.expand_dims(phase, 4)

print("fit model")
history = model.fit(train_images, train_labels,  epochs=dataset_iterations, batch_size=batch_size, shuffle=True,
          callbacks = [cp_callback,earlystop])  # pass callback to training for saving the model80

loss_historyGA = history.history['loss']


with open('loss_historyGA.pickle', 'wb') as f:
    pickle.dump([loss_historyGA, dataset_iterations], f)
    

##
###################

#save model
if not os.path.exists("models/backgroundremoval"): 
    os.makedirs("models/backgroundremoval") 
    
model_name = "models/backgroundremoval/modelBR_trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + "_loss_"+ lossU +".h5"
model.save(model_name)


#tf.keras.models.save_model(model, model_name)    
    
    
###################


    
#################################
#plot loss

if not os.path.exists("models/backgroundremoval/loss"): 
    os.makedirs("models/backgroundremoval/loss") 
    
plt.figure(figsize=(6, 3))
plt.plot(loss_historyGA)
plt.ylim([0, loss_historyGA[-1]*2])
plt.title("Loss")
plt.xlabel("Dataset iterations")
lossnamefile = "models/backgroundremoval/loss/modelBR_trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ lossU
lossfile_extension =".png"
plt.savefig(lossnamefile + lossfile_extension )

###############
# plot loss as txt
lossfile_extension =".txt"

file = open(lossnamefile+lossfile_extension,'w')
for item in loss_historyGA:

	file.write(str(item)+"\n")
file.close()