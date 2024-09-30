"""
Created on Mon Feb 19 09:18:00 2024

@author: catarinalopesdias
trains 1 Unet network 
"""
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from keras.layers import Input, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pickle

from newGA import GradientAccumulateModel
from  my_classes.dataGenerator.unif02.DataGenerator_susc02_RectCirc_gt import DataGeneratorUniform_RecCirc_gt
from  my_classes.dataGenerator.unif02.DataGenerator_susc02_RectCirc_gt import DataGeneratorUniform_RecCirc_gt


######################
# create training set 
######################

num_train_instances = 1000

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
text_susc="unif02_RecCirc_"
training_generatorUnif   = DataGeneratorUniform_RecCirc_gt(partition['train'])
validation_generatorUnif = DataGeneratorUniform_RecCirc_gt(partition['validation'])

########################################################################################################
########################################################################################################

#############################################################
#### Compile the model
#############################################################


#from networks.network_catarina_new import build_CNN_catarina_inputoutput
#from networks.network_catarina_new4convs3levels import build_CNN_catarina_inputoutput
from networks.network_Heber_new4convs4levels import build_CNN_Heber_inputoutput
#from networks.network_Heber_new4convs3levels import build_CNN_Heber_inputoutput

nettype = "4convs4levels_BgfRemov"

#preprocessinglayers
#from my_classes.keraslayer.layerbackgroundfield_artifacts_nowrapping import CreatebackgroundFieldLayer
#from my_classes.keraslayer.layerbackgroundfield_artifacts_wrapping import CreatebackgroundFieldLayer
from my_classes.keraslayer.layerbackgroundfield_artifacts_final import CreatebackgroundFieldLayer

from my_classes.keraslayer.layer_phase import CreatePhaseLayer
#from my_classes.keraslayer.layer_mask_inputtensor import CreateMaskLayer
from my_classes.keraslayer.layer_mask import CreateMaskLayer


#old - original seems to work
input_shape = (128, 128, 128, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")


###################################
#create layers
LayerwMask = CreateMaskLayer()
LayerWBackgroundField = CreatebackgroundFieldLayer()
LayerWPhase = CreatePhaseLayer()

bgf_features = "zgradient_no_boundartif_yes_wrap_yes_"
#################################################
#################################################
# phase #inputs is gt #######

x_phase = LayerWPhase(input_tensor) 
# mask
x_mask = LayerwMask(x_phase) 
#x_mask = LayerwMask(input_tensor) 

mask = x_mask[0]
maskedPhase = x_mask[1]
#backgroundfield
phasewithbackgroundfield = LayerWBackgroundField(x_mask)
#bgfremoval_phase= build_CNN_Heber_inputoutput4convs4levels(phasewithbackgroundfield)
#dipinv_gt= build_CNN_Heber_inputoutput4convs3levels(bgfremoval_phase)

y = tf.keras.layers.Concatenate()([maskedPhase, build_CNN_Heber_inputoutput(phasewithbackgroundfield)])
##########################################

outputs = [phasewithbackgroundfield,
           y]

#outputs = [phasewithbackgroundfield,y,dipinv_gt]


model = Model(input_tensor, outputs)

name = "PhaseBgf_BgfRem" + nettype + bgf_features


model.summary() 
###############################################
###############################################

print("Model with gradient accumulation")
gaaccumsteps = 10
#learningrate
lr =0.0001
text_lr = str(lr).split(".")[1]

model = GradientAccumulateModel(accum_steps=gaaccumsteps,
                                inputs=model.input, outputs=model.output)


#optimizer
optimizerMINE = Adam(
              learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
               epsilon=1e-8
               ) 

##################################################################################
####################################################################################

#### loss ##################

###############################################################################
### Add costum loss
import keras.backend as K

lossU = "costum"# "mse" #
###############################

def my_loss_function(y_true, y_pred):
    print("----------loss function")
    
    #print("y_pred shape")
    
    #print(y_pred.shape)
    #print(K.int_shape(y_pred))

    #print(y_pred[0])
    maskedPhaseTrue = y_pred[:,:,:,:,0]
    maskedPhaseTrue= tf.expand_dims(maskedPhaseTrue, 4)

    #print("true phase data shape")
    #print(maskedPhaseTrue.shape)
    #print("second element")
    
    # y pred
    maskedPhasePrediction = y_pred[:,:,:,:,1]
    maskedPhasePrediction = tf.expand_dims(maskedPhasePrediction, 4)
    #print("masked prediction shape")
    #print(maskedPhasePrediction.shape)
    
    # image reconstruction
    #image_loss = tf.keras.losses.MSE(maskedPhaseTrue, maskedPhasePrediction)
    mse_loss = tf.keras.losses.MeanSquaredError()
    image_loss = mse_loss(maskedPhaseTrue, maskedPhasePrediction)
    #print("loss shape")
    #print(image_loss)
    #print(image_loss.shape)

    return  image_loss #image_loss
###################################################################


model.compile(optimizer=optimizerMINE, loss=[None, my_loss_function])

#model.compile(optimizer=optimizerMINE, loss=[None, my_loss_function,"mse"])


model.summary()
#########################################################

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

dataset_iterations = 3000
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
    text_lr +"_"+ lossmon+"_" + text_susc + "_phasemaskbgf_bgfremovalheber_datagenunif02rectcircles_wrap.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only = False,
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
    start_from_epoch=100,
) 




print("fit model")

history = model.fit( x=training_generatorUnif,
                    validation_data=validation_generatorUnif,
                    epochs=dataset_iterations,
                    use_multiprocessing=True,
                    initial_epoch=905,
                    callbacks = [cp_callback, earlystop],
                    workers=6)




val_loss_historyGA = history.history['val_loss']
loss_historyGA = history.history['loss']


#save model
if not os.path.exists("models/preprocessing_backgroundremoval"):
    os.makedirs("models/preprocessing_Backgroundremoval")

    
model_name1 = "models/preprocessing_backgroundremoval/model_Prep_BR_" + name + \
"_newadam_" + str(num_filter)+"filters_trainsamples" + str(num_train_instances) + \
"_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + \
"_loss_" + lossU + \
"_" + text_lr + "_" + lossmon + "_" + text_susc + "_phasemaskbgf_bgfremovalheber_datagenunif02rectcircles_wrap.keras"


model.save(model_name1)



#################################
#plot loss

lossfile_extensionpng =".png"
lossfile_extensiontxt =".txt"

if not os.path.exists("models/preprocessing_backgroundremoval/loss"):
    os.makedirs("models/preprocessing_backgroundremoval/loss")

plt.figure(figsize=(6, 3))
plt.plot(loss_historyGA)
plt.ylim([0, loss_historyGA[-1]*5])
plt.title("loss")
plt.xlabel("Dataset iterations")
lossnamefile = "models/preprocessing_backgroundremoval/loss/model_Prep_BR_" + name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + lossU + "_" + \
text_lr + "_" + "loss"+"_"+text_susc+"_phasemaskbgf_bgfremovalheber_datagenunif02rectcircles_wrap"
plt.savefig(lossnamefile + lossfile_extensionpng )
##################################
plt.figure(figsize=(6, 3))
plt.plot(val_loss_historyGA)
plt.title(lossmon)
plt.xlabel("Dataset iterations")
vallossnamefile = "models/preprocessing_backgroundremoval/loss/model_Prep_BR_" + name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + lossU + "_" + \
text_lr + "_" + lossmon+"_"+text_susc+"_phasemaskbgf_bgfremovalheber_datagenunif02rectcircles_wrap"
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
