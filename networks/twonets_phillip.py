from keras.layers import Input ,Conv3D, Conv3DTranspose, LeakyReLU, UpSampling3D, Concatenate , Add, BatchNormalization
from keras.models import Model
from keras.initializers import GlorotNormal
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow.keras.models import save_model
from datahandling import read_and_decode_tf, visualize_all


# Check https://colab.research.google.com/drive/1Omj-taD4P4oBBZOrZKf8gdfp8sMP972r?usp=sharing

###########################################
#build convolutional network
def build_CNN(input_tensor):


#########################################################################################################################
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(input_tensor)
 
    output_tensor=residual_conn

    return output_tensor


  ####################################################
  ##    Process data for training and define hyperparameters
  ####################################################
t
#############################################################
#### Compile the model
#############################################################
input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")

ushape1 = build_CNN(input_tensor)
ushape2 = build_CNN(ushape1)

model = Model(input_tensor, ushape2) #get from orig deepQSM algorithm

model.compile(optimizer='adam', loss='mean_absolute_error')


#
#################################################################
### Train the model
#################################################################



checkpoint_path1 = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir1 = os.path.dirname(checkpoint_path1)

cp_callback = ModelCheckpoint(checkpoint_path1,
                                save_weights_only = True,
                                save_freq = 10,
                                verbose = 1)

steps_p_e = int(len(tfrecord_files_train)//batch_size)
print("steps_p_epoch", steps_p_e)

dataset_train_np = tfrecord_dataset_train.as_numpy_iterator()

model_train = model.fit(data_generator(train_data=dataset_train_np),
                        steps_per_epoch=steps_p_e, epochs=num_epochs, 
                        shuffle=True, callbacks = [cp_callback]) # AHHHHHHHHHHHHHHHHHH initial_epoch=61

loss_history1 = model_train.history['loss']

with open('loss_history1.pickle', 'wb') as f:
    pickle.dump([loss_history1, num_epochs], f)

# if the demo_folder directory is not present then create it. 
if not os.path.exists("models"): 
    os.makedirs("models") 
    
model_name = "models/model_" + str(num_epochs) +"epochs_" + "batchsize"+ str(batch_size) + "_trnfolder_" + folder_trn+".h5"
#model.save(model_name)
save_model(model, model_name)
########################################################################
### Load latest checkpoints from training
########################################################################

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

input_shape = (None, None, None, 1)
input_tensor = Input(shape= input_shape, name="input")
ushape1 = build_CNN(input_tensor)
ushape2 = build_CNN(ushape1)
model = Model(input_tensor, ushape2)

model.compile(optimizer='adam', loss='mean_absolute_error')

latest_weights = tf.train.latest_checkpoint(checkpoint_dir)
print(latest_weights)
model.load_weights(latest_weights)


##################################################################
############ Predict and plot original data
#################################################################
dataset_test_np = parsed_dataset_tst.as_numpy_iterator()

counter = 0
for data_sample_tst in dataset_test_np:
    input_ds_tst, output_ds_tst = data_sample_tst
    
    X_test = tf.expand_dims(input_ds_tst, 0) #batch dimension
    X_test = tf.expand_dims(X_test, 4)  # channel dimension
    
    predicted = model.predict(X_test)

    counter+=1
    print(f"Test set {counter}")
    #visualize(X_test[0,:,:,:,0], para="input")
    #visualize(output_d, para="ref")
    #visualize(predicted[0,:,:,:,0],para="pred")
    
    #prediction_title ="prediction_batch" + str(batch_size) + "_" + str(num_epochs) + "epochs_trn_" folder_trn +"tst" + 
    prediction_title ="prediction_batch" + str(batch_size) + "_" + str(num_epochs) + "epochs_" + folder_trn +"_"+ folder_test + "_" + str(counter)
    print(prediction_title)
    visualize_all(X_test[0,:,:,:,0], output_ds_tst, predicted[0,:,:,:,0], prediction_title)
  # for plotting just the first image - uncomment if you want to see every prediction
    #break

#x_axis = np.linspace(1,num_epochs,num_epochs)
x_axis = np.linspace(61,num_epochs,num=num_epochs-61)
a = tf.linspace(1,60,60)
plt.plot(a,a)

plt.plot(loss_history1)
plt.ylabel('Loss')
plt.xlabel('Epochs')
#plt.yscale('log')
plt.ylim(0, 0.25) 
plt.xlim(0, num_epochs) 
plt.title("Loss history")
plt.ticklabel_format(style='plain')

plt.plot([0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4])


loss_title ="images/loss_batch" + str(batch_size) + "_" + str(num_epochs) + "epochs_" + folder_trn +"_"+ folder_test 
plt.savefig(loss_title)
plt.show()

