"""
Created on Wed Dec  6 09:28:17 2023

@author: catarinalopesdias
trains 2 Unets networks
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
from  my_classes.DataGenerator_susc02_doublenet import DataGeneratorUniform

from networks.network_phillip_inputoutput import build_CNN_phillip_inputoutput
from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput





num_train_instances = 500
samples_dic = []


for i in range(num_train_instances):

    samples_dic.append( str(i) + 'samples' )
   

#######################
partition_factor = 0.8

partition = {'train': samples_dic[0: int(partition_factor * num_train_instances)] , 
             'validation': 
             samples_dic[ -int((1-partition_factor) * num_train_instances): num_train_instances]}


# Generators
#text regarding susc
text_susc="unif02"
training_generator   = DataGeneratorUniform(partition['train'])
validation_generator = DataGeneratorUniform(partition['validation'])





#############################################################
#### Compile the model
#############################################################
input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")

print("compile model")

ushape1 = build_CNN_BOLLMANinputoutput(input_tensor)
ushape2 = build_CNN_phillip_inputoutput(tf.keras.backend.stop_gradient(ushape1))  #ushape1 ---- tf.stop_gradient(ushape1) tf.keras.backend.stop_gradient
model = Model(input_tensor, outputs={'ushape1': ushape1, 'ushape2': ushape2}) #maybe ushape2.input -- [ushape1,ushape2] ----  [ushape1]
name = "BollmannPhillip"

#using module class
#model1 = build_CNN_BOLLMAN(input_tensor)
#model2 = build_CNN_BOLLMAN(model1.output)
#full_model  = Model(inputs=new_model.input, outputs=output)
#model = Model(input_tensor, model2) 


###############################################
###############################################
print("Model with gradient accumulation")
gaaccumsteps = 10;
#learningrate
lr =0.0005
text_lr = str(lr).split(".")[1]
text_stop = "stopping"

model = GradientAccumulateModel(accum_steps=gaaccumsteps, 
                                inputs=model.input, outputs=model.outputs)

loss_model1 = "mse"    #mse "mean_absolute_error"
loss_model2 = "mse"    #mse "mean_absolute_error"
losses = [loss_model1, loss_model2]

#optimizer
optimizerMINE = Adam(
              learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
               epsilon=1e-8
               ) 



model.compile(optimizer=optimizerMINE, loss = losses)

model.summary()

##########################################################################
# training part
##########################################################################
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
batch_size = 1
num_filter = 16
lossmon = "val_loss"
###############
# "cp-{epoch:04d}"+

checkpoint_path = "checkpoints/doublenet/DN_"+\
    name + "_newadam" + str(num_filter)+ \
    "_trainsamples" + str(num_train_instances) + \
    "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ \
    "_gaaccum" + str(gaaccumsteps) + \
    "_loss_" + losses[0] +losses[1]+ "_" + text_stop+ \
    text_lr +"_"+ lossmon+"_" + text_susc + "_datagen.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only = False,
                                                 #save_freq=save_period,
                                                 save_freq="epoch",
                                                 save_best_only=True,
                                                 monitor = lossmon,
                                                 verbose=1)



earlystop = tf.keras.callbacks.EarlyStopping(
    monitor=lossmon,
    min_delta=0,
    patience=100,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)




print("fit model")

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=dataset_iterations,
                    use_multiprocessing=True,
                    #batch_size=1, 
                    callbacks = [cp_callback,earlystop],
                    workers=6)




loss_historyGA = history.history['loss']


#with open('loss_historyGA.pickle', 'wb') as f:
#pickle.dump([loss_historyGA, dataset_iterations], f)
###################

#save model
if not os.path.exists("models/doublenet"):
    os.makedirs("models/doublenet")

model_name1 = "models/doublenet/model_DN_" + name + \
"_newadam_" + str(num_filter)+"filters_trainsamples" + str(num_train_instances) + \
"_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + \
"_loss_" + losses[0] +losses[1] + text_stop + \
"_" + text_lr + "_"+ lossmon + "_" + text_susc + "_datagen.keras"


model.save(model_name1)



#################################
#plot loss

lossfile_extensionpng =".png"
lossfile_extensiontxt =".txt"

if not os.path.exists("models/doublenet/loss"):
    os.makedirs("models/doublenet/loss")

plt.figure(figsize=(6, 3))
plt.plot(loss_historyGA)
#plt.ylim([0, loss_historyGA[-1]*2])
plt.title("Loss")
plt.xlabel("Dataset iterations")
lossnamefile = "models/doublenet/loss/model_DB_" + name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + losses[0] +losses[1] + "_" + \
text_lr + "_" + lossmon+"_"+text_susc+"_datagen"
plt.savefig(lossnamefile + lossfile_extensionpng )

###############
# plot loss as txt

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

lossnamefileF = "models/doublenet/loss/model_DB_" + name + "_1_"\
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + losses[0] +losses[1] + "_" + \
text_lr + "_" + lossmon+"_"+text_susc+"_datagen"



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
lossnamefileF = "models/doublenet/loss/model_DB_" + name + "_2_"\
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + losses[0] +losses[1] + "_" + \
text_lr + "_" + lossmon+"_"+text_susc+"_datagen"


plt.savefig(lossnamefileS + lossfile_extensionpng )

lossfile_extension =".txt"

file = open(lossnamefileS + lossfile_extensiontxt,'w')
for item in loss_historyS:

	file.write(str(item)+"\n")
file.close()

