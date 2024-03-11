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

import os
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle

from newGA import GradientAccumulateModel
from networks.network_phillip import build_CNN
#from networks.network_BOLLMAN import build_CNN_BOLLMAN
from networks.network_adaptedfrom_BOLLMAN import build_CNN_BOLLMAN
from visualize_volumes import view_slices_3dNew

################################################
#   Import data
################################################
#view input
#view output

#phase_bg = np.load('syntheticdata/phase_bg100.npy')
#phase = np.load('syntheticdata/phase100.npy')
######
# compressed data
loaded = np.load('datasynthetic/150samples.npz')

phase = loaded['phase1']
gt = loaded['sim_gt1']
del loaded
#######################


######
# compressed data
# loaded1= np.load('datasynthetic/150samples_1.npz')

#phase1 = loaded1['phase1']
#phase_bg1 = loaded1['phase_bg1']
#del loaded1

#phase = np.concatenate((phase, phase1))
#phase_bg = np.concatenate((phase_bg, phase_bg1))

# del phase_bg1, phase1

######################################
num_slice = 3


view_slices_3dNew(phase[num_slice,:,:,:], 50, 50,50, vmin=-10, vmax=10, title="phase") 
view_slices_3dNew(gt[num_slice,:,:,:], 50, 50,50, vmin=-10, vmax=10, title="gt")
del num_slice

num_train_instances = phase.shape[0]

##############################


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
#model = build_CNN_BOLLMAN(input_tensor)

#model.compile(optimizer='adam', loss='mean_absolute_error')
#model.summary()

###############################################
###############################################
print("Model with gradient accumulation")
gaaccumsteps = 10;
model = GradientAccumulateModel(accum_steps=gaaccumsteps, inputs=model.input, outputs=model.output)

# we changed to mean absolute error
lossU = "mse" #"mean_absolute_error"#"mse"# "mean_absolute_error" #"mse"    #mse
#opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9 ,beta_2 =0.999, epsilon=1e-8, gradient_accumulation_steps=gaaccumsteps )

# from model manager
optimizerMINE = Adam(
              #learning_rate=0.0003,
                beta_1=0.9,
                beta_2=0.999,
               epsilon=1e-8
               ) #gradient_accumulation_steps=10


 #


model.compile(optimizer=optimizerMINE, loss= lossU) #mean_absolute_error

#model.compile(optimizer='adam', loss= lossU) #mean_absolute_error
model.summary()

#################################################################################

###############################################################################
###############################################################################
print("untrained")
# what does the untrained model predict
test_epoch_nbr = 3
X_test = phase[np.newaxis, test_epoch_nbr,:,:,:, np.newaxis]
print(X_test.shape)

y_pred = model.predict(X_test)

print(y_pred.shape)


#view_slices_3dNew(phase_bg[test_epoch_nbr, :, :, :], 50,50,50, vmin=-1, vmax=1, title=' Phase and background')
#view_slices_3dNew(phase[test_epoch_nbr, :, :, :],50,50,50 , vmin=-1, vmax=1, title='Phase')
#view_slices_3dNew(y_pred[0, :, :, :, 0], 50,50,50, vmin=-1, vmax=1, title='Predicted phase')
del y_pred, test_epoch_nbr


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

dataset_iterations = 5000
save_period = 150
batch_size = 1
num_filter = 16
###############
# {epoch:04d}
checkpoint_path = "checkpoints/dipoleinversion/newadam_" + str(num_filter)+"filter"+ "_trainsamples" + str(num_train_instances) + "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU+".ckpt"

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
    monitor="loss",
    min_delta=0,
    patience=100,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)




train_images =tf.expand_dims(phase, 4)
train_labels = tf.expand_dims(gt, 4)

print("fit model")
history = model.fit(train_images, train_labels,  epochs=dataset_iterations, batch_size=batch_size, shuffle=True,#nitial_epoch = 289,
          callbacks = [cp_callback,earlystop])  # pass callback to training for saving the model80

loss_historyGA = history.history['loss']


with open('loss_historyGA.pickle', 'wb') as f:
    pickle.dump([loss_historyGA, dataset_iterations], f)
    

##
###################

#save model
#if not os.path.exists("models/backgroundremoval"): 
#    os.makedirs("models/backgroundremoval") 
    
if not os.path.exists("models/dipoleinversion"): 
    os.makedirs("models/dipoleinversion")     
    
model_name1 = "models/dipoleinversion/model_newadam_" + str(num_filter)+"filters_trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + "_loss_"+ lossU +".keras"


#model.save(model_name)

model.save(model_name1)
#tf.keras.models.save_model(model, model_name)    
    
    
###################


    
#################################
#plot loss

lossfile_extensionpng =".png"
lossfile_extensiontxt =".txt"

if not os.path.exists("models/dipoleinversion/loss"): 
    os.makedirs("models/dipoleinversion/loss") 
    
plt.figure(figsize=(6, 3))
plt.plot(loss_historyGA)
#plt.ylim([0, loss_historyGA[-1]*2])
plt.title("Loss")
plt.xlabel("Dataset iterations")
lossnamefile = "models/dipoleinversion/loss/modelBR_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) + "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ "_gaaccum"+ str(gaaccumsteps) +"_loss_"+ lossU
plt.savefig(lossnamefile + lossfile_extensionpng )

###############
# plot loss as txt

file = open(lossnamefile+lossfile_extensiontxt,'w')
for item in loss_historyGA:

	file.write(str(item)+"\n")
file.close()