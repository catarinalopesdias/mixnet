"""
Created on Mon Feb 19 09:18:00 2024

@author: catarinalopesdias
trains 1 Unet network 
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
from  my_classes.DataGenerator_susc02_dipinv import DataGeneratorUniform
from networks.network_phillip import build_CNN_phillip


#from networks.network_phillip_inputoutput import build_CNN_phillip_inputoutput
#from networks.network_BOLLMAN import build_CNN_BOLLMAN
#from networks.network_adaptedfrom_BOLLMAN import build_CNN_BOLLMAN

num_train_instances = 500
samples_dic = []


for i in range(num_train_instances):

    samples_dic.append( str(i) + 'samples' )
   

#######################
partition_factor = 0.8

partition = {'train': samples_dic[0: int(partition_factor * num_train_instances)] , 
             'validation': 
             samples_dic[ -int(( 1-partition_factor) * num_train_instances): num_train_instances]}


# Generators
#text regarding susc
text_susc="unif02"
training_generatorUni   = DataGeneratorUniform(partition['train'])
validation_generatorUni = DataGeneratorUniform(partition['validation'])




#############################################################
#### Compile the model
#############################################################
input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")

print("compile model")

model = build_CNN_phillip(input_tensor)
name = "Phillip"

###############################################
###############################################
print("Model with gradient accumulation")
gaaccumsteps = 10;
#learningrate
lr =0.0005
text_lr = str(lr).split(".")[1]

model = GradientAccumulateModel(accum_steps=gaaccumsteps,
                                inputs=model.input, outputs=model.output)

lossU = "mse" #"mean_absolute_error"#"mse"# "mean_absolute_error" #"mse"    #mse

#optimizer
optimizerMINE = Adam(
              learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
               epsilon=1e-8
               ) 



model.compile(optimizer=optimizerMINE, loss = lossU) #mean_absolute_error

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

checkpoint_path = "checkpoints/dipoleinversion/DipInv_" +\
    name + "_newadam" + str(num_filter)+ \
    "_trainsamples" + str(num_train_instances) + \
    "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ \
    "_gaaccum" + str(gaaccumsteps) + \
    "_loss_" + lossU + "_" + \
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
    patience=50,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)




print("fit model")

history = model.fit_generator(generator=training_generatorUni,
                    validation_data=validation_generatorUni,
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
if not os.path.exists("models/dipoleinversion"):
    os.makedirs("models/dipoleinversion")

    
model_name1 = "models/dipoleinversion/model_DipInv_" + name + \
"_newadam_" + str(num_filter)+"filters_trainsamples" + str(num_train_instances) + \
"_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + \
"_loss_" + lossU + \
"_" + text_lr + "_" + lossmon + "_" + text_susc + "_datagen.keras"


model.save(model_name1)



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
lossnamefile = "models/dipoleinversion/loss/model_DipInv_" + name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + lossU + "_" + \
text_lr + "_" + lossmon+"_"+text_susc+"_datagen"
plt.savefig(lossnamefile + lossfile_extensionpng )

###############
# plot loss as txt

file = open(lossnamefile + lossfile_extensiontxt,'w')
for item in loss_historyGA:

	file.write(str(item)+"\n")
file.close()