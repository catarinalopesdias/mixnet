"""
Created on Mon Feb 19 09:18:00 2024

@author: catarinalopesdias
trains 1 Unet network 
"""
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from keras.layers import Input, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle

from newGA import GradientAccumulateModel
from  my_classes.dataGenerator.DataGenerator_susc02_bgrem_RectCirc_evenlessbgnoartifacts_onlyphaseExtraLayer1output import DataGeneratorUniformevenlessbgnoartifactsExtraLayer


######################
# create training set 
######################

num_train_instances = 500

# validation set 
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
training_generatorUnif   = DataGeneratorUniformevenlessbgnoartifactsExtraLayer(partition['train'])
validation_generatorUnif = DataGeneratorUniformevenlessbgnoartifactsExtraLayer(partition['validation'])

########################################################################################################
########################################################################################################
########################################################################################################

#############################################################
#### Compile the model
#############################################################

# HEREEEEEEEEEEE
from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
from my_classes.keraslayer.layerbackgroundfield_artifacts_nowrapping_nodim import CreatebackgroundFieldLayer
from my_classes.keraslayer.layer_mask import CreateMaskLayer



#old - original seems to work
#input_shape = (None, None, None, 1) # shape of pict, last is the channel
#input_shape = (128, 128, 128, 1) # shape of pict, last is the channel

 #inputs = [Input(shape=input_shape), Input(shape=input_shape), Input(shape=(1,))]
 #inputs = [Input(shape=input_shape), Input(shape=(1,))] # Gets phase 
 
 
#input_tensor = Input(shape = input_shape, name="input") #phase!

#inputs = [Input(shape=input_shape), Input(shape=input_shape)]


#input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_shape = (128, 128, 128, 1) # shape of pict, last is the channel

#inputs = [Input(shape=input_shape), Input(shape=(1,)) ] # Gets phase (and extra)
inputs = [Input(shape=input_shape) ] # Gets phase (and extra)

###################################
#create layers
LayerwMask = CreateMaskLayer()
LayerWBackgroundField = CreatebackgroundFieldLayer()
########################################################
#################################################
#
# mask

x = LayerwMask(inputs) #inputs is phase #######and shape

mask, phasewithMask = x

#backgroundfield
phasewithbackgroundfield = LayerWBackgroundField(LayerwMask(inputs))


##########################################

#outputs = build_CNN_BOLLMANinputoutput(x)
outputs = build_CNN_BOLLMANinputoutput(phasewithbackgroundfield)
#outputs = build_CNN_BOLLMANinputoutput(LayerWBackgroundField(LayerwMask(inputs)))

model = Model(inputs, outputs)
name = "Preprocess_Bollmann"
model.summary()




###############################################
###############################################
print("Model with gradient accumulation")
gaaccumsteps = 10
#learningrate
lr =0.001 #0.0004
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



#model.compile(optimizer=optimizerMINE, loss = lossU,run_eagerly=True)
model.compile(optimizer=optimizerMINE, loss = lossU)

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

checkpoint_path = "checkpoints/preprocessing_bgremovalmodel/Bg_" +\
    name + "_newadam" + str(num_filter)+  "cp-{epoch:04d}"\
    "_trainsamples" + str(num_train_instances) + \
    "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ \
    "_gaaccum" + str(gaaccumsteps) + \
    "_loss_" + lossU + "_" + \
    text_lr +"_"+ lossmon+"_" + text_susc + "_datagen_evenlessbgn.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only = False,
                                                 #save_freq=save_period,
                                                 save_freq="epoch",
                                                 save_best_only=False,
                                                 monitor = lossmon,
                                                 verbose=1)



earlystop = tf.keras.callbacks.EarlyStopping(
    monitor=lossmon,
    min_delta=0,
    patience=1000,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
) 




print("fit model")

history = model.fit( x=training_generatorUnif,
                    validation_data=validation_generatorUnif,
                    epochs=dataset_iterations,
                    use_multiprocessing=True,
                    callbacks = [cp_callback, earlystop],
                    workers=6)




val_loss_historyGA = history.history['val_loss']
loss_historyGA = history.history['loss']




#save model
if not os.path.exists("models/preprocessing_Backgroundremoval"):
    os.makedirs("models/preprocessing_Backgroundremoval")

    
model_name1 = "models/preprocessing_Backgroundremoval/model_Prep_BR_" + name + \
"_newadam_" + str(num_filter)+"filters_trainsamples" + str(num_train_instances) + \
"_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + \
"_loss_" + lossU + \
"_" + text_lr + "_" + lossmon + "_" + text_susc + "_datagen_evenlessbgnoartifacts_Circ.keras"


model.save(model_name1)



#################################
#plot loss

lossfile_extensionpng =".png"
lossfile_extensiontxt =".txt"

if not os.path.exists("models/preprocessing_Backgroundremoval/loss"):
    os.makedirs("models/preprocessing_Backgroundremoval/loss")

plt.figure(figsize=(6, 3))
plt.plot(loss_historyGA)
#plt.ylim([0, loss_historyGA[-1]*2])
plt.title("loss")
plt.xlabel("Dataset iterations")
lossnamefile = "models/preprocessing_Backgroundremoval/loss/model_Prep_BR_" + name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + lossU + "_" + \
text_lr + "_" + "loss"+"_"+text_susc+"_datagen_evenlessbgnoartifacts_artif_nowrappingCircEager"
plt.savefig(lossnamefile + lossfile_extensionpng )
##################################
plt.figure(figsize=(6, 3))
plt.plot(val_loss_historyGA)
#plt.ylim([0, loss_historyGA[-1]*2])
plt.title(lossmon)
plt.xlabel("Dataset iterations")
vallossnamefile = "models/backgroundremovalBOLLMAN_ExtraLayer/loss/model_Prep_BR_" + name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + lossU + "_" + \
text_lr + "_" + lossmon+"_"+text_susc+"_datagen_evenlessbg_artifacts__artif_nowrappingCirc"
plt.savefig(vallossnamefile + lossfile_extensionpng )

###############
# plot loss as txt

file = open(lossnamefile + lossfile_extensiontxt,'w')
for item in loss_historyGA:

	file.write(str(item)+"\n")
file.close()

############

file = open(vallossnamefile + lossfile_extensiontxt,'w')
for item in val_loss_historyGA:

	file.write(str(item)+"\n")
file.close()
