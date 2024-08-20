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
from  my_classes.dataGenerator.DataGenerator_susc02_RectCirc_phase import DataGeneratorUniform_RecCirc_phase


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
text_susc="unif02_RecCirc_"
training_generatorUnif   = DataGeneratorUniform_RecCirc_phase(partition['train'])
validation_generatorUnif = DataGeneratorUniform_RecCirc_phase(partition['validation'])

########################################################################################################
########################################################################################################

#############################################################
#### Compile the model
#############################################################

# HEREEEEEEEEEEE
from networks.network_adaptedfrom_BOLLMAN_inputoutput import build_CNN_BOLLMANinputoutput
#preprocessinglayers
from my_classes.keraslayer.layerbackgroundfield_artifacts_nowrapping import CreatebackgroundFieldLayer
#from my_classes.keraslayer.layer_mask import CreateMaskLayer
from my_classes.keraslayer.layer_mask_inputtensor import CreateMaskLayer


#old - original seems to work
input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_shape = (128, 128, 128, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")

#inputs = [Input(shape=input_shape) ] # Gets only phase

#inputs = [Input(shape=input_shape, name="inputs_0") ] # Gets only phase
#inputs = [input_tensor] # Gets only phase


###################################
#create layers
LayerwMask = CreateMaskLayer()
LayerWBackgroundField = CreatebackgroundFieldLayer()
#################################################
#################################################
# mask

#mask, maskedPhase = LayerwMask(inputs) #inputs is phase #######and shape
#mask, maskedPhase = LayerwMask(input_tensor) #inputs is phase #######and shape
x = LayerwMask(input_tensor) #inputs is phase #######and shape
mask = x[0]
maskedPhase = x[1]
#backgroundfield
#phasewithbackgroundfield = LayerWBackgroundField(LayerwMask(inputs))
phasewithbackgroundfield = LayerWBackgroundField(x)


#phasewithbackgroundfield = LayerWBackgroundField(LayerwMask(input_tensor))
y = tf.keras.layers.Concatenate()([maskedPhase, build_CNN_BOLLMANinputoutput(phasewithbackgroundfield)])

##########################################
#outputs = build_CNN_BOLLMANinputoutput(x)
#outputs = build_CNN_BOLLMANinputoutput(LayerWBackgroundField(LayerwMask(inputs)))

outputs = [#mask,
           #maskedPhase,
           phasewithbackgroundfield,y]

#model = Model(input=inputs, output=outputs)
#model = Model(inputs, outputs)
model = Model(input_tensor, outputs)

name = "PhaseBgf_Bgfrem_Bollmann"



###############################################
###############################################

print("Model with gradient accumulation")
gaaccumsteps = 10
#learningrate
lr =0.0005#0.001 #0.0004
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

lossU = "costum"# "mse" #"mean_absolute_error"#"mse"# "mean_absolute_error" #"mse"    #mse
#losses = [None, None, lossU]


###############################################################################
### Add costum loss
import keras.backend as K

###############################

def my_loss_function(y_true, y_pred):
    print("----------loss function")
    print("y_pred shape")
    print(y_pred.shape)
    #print(y_pred[0])
    maskedPhaseOutput = y_pred[:,:,:,:,0]
    maskedPhaseOutput = tf.expand_dims(maskedPhaseOutput, 4)
   #maskedPhaseOutput = y_pred[0]
    print("phase data shape")
    print(maskedPhaseOutput.shape)
    #print("second element")
    
    maskedPhasePrediction = y_pred[:,:,:,:,1]
  

    maskedPhasePrediction = tf.expand_dims(maskedPhasePrediction, 4)
    print("pred shape")
    print(maskedPhasePrediction.shape)
    
    #print(y_pred[1])
    #masked, bgf , predphase = y_pred

    #plt.imshow(pred_phase[0,64,:,:,0 ], cmap='gray',  vmin=-0.4, vmax=0.4)   
    #plt.show()
    #print("y_true")
    #print(y_true)
    #print(y_true.shape)
    #print("y_true second element")
    #print(y_true[1])
    #print("y_true third element")
    #print(y_true[2])
    #print("y_true fourth element")
    #print(y_true[3])
    # image reconstruction
    image_loss = tf.keras.losses.mse(maskedPhaseOutput, maskedPhasePrediction)
    print("loss shape")
    print(image_loss.shape)
    #image_loss =  tf.keras.losses.MeanSquaredError(maskedPhaseOutput, maskedPhasePrediction)
    #bla=maskedPhaseOutput[0, 64,:,:]
    #plt.imshow(bla.numpy(), cmap='gray',  vmin=-0.1, vmax=0.1)   
    #plt.show()
    #image_loss =  K.mean(K.square(maskedPhaseOutput - maskedPhasePrediction))

    #print("image loss shape")
    #print(image_loss)
    #print(image_loss.shape)
    return  image_loss #image_loss
###################################################################


#model.compile(optimizer=optimizerMINE, loss = lossU,run_eagerly=True)
#model.compile(optimizer=optimizerMINE, loss = lossU)
#model.compile(optimizer="Adam", loss = losses)
#model.compile(optimizer="Adam", loss = ["mse","mse","mse" ])
#model.compile(optimizer=optimizerMINE, loss=my_loss_function)
#model.compile(optimizer="Adam", loss=[None, my_loss_function])


model.compile(optimizer=optimizerMINE, loss=[None, my_loss_function])


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
    text_lr +"_"+ lossmon+"_" + text_susc + "_datagenRecCircNewLoss.ckpt"

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
    patience=10000,
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
if not os.path.exists("models/preprocessing_backgroundremoval"):
    os.makedirs("models/preprocessing_Backgroundremoval")

    
model_name1 = "models/preprocessing_backgroundremoval/model_Prep_BR_" + name + \
"_newadam_" + str(num_filter)+"filters_trainsamples" + str(num_train_instances) + \
"_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + \
"_loss_" + lossU + \
"_" + text_lr + "_" + lossmon + "_" + text_susc + "_datagenRecCircNewLoss.keras"


model.save(model_name1)



#################################
#plot loss

lossfile_extensionpng =".png"
lossfile_extensiontxt =".txt"

if not os.path.exists("models/preprocessing_backgroundremoval/loss"):
    os.makedirs("models/preprocessing_backgroundremoval/loss")

plt.figure(figsize=(6, 3))
plt.plot(loss_historyGA)
#plt.ylim([0, loss_historyGA[-1]*2])
plt.title("loss")
plt.xlabel("Dataset iterations")
lossnamefile = "models/preprocessing_backgroundremoval/loss/model_Prep_BR_" + name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + lossU + "_" + \
text_lr + "_" + "loss"+"_"+text_susc+"_datagenRecCircNewLoss"
plt.savefig(lossnamefile + lossfile_extensionpng )
##################################
plt.figure(figsize=(6, 3))
plt.plot(val_loss_historyGA)
#plt.ylim([0, loss_historyGA[-1]*2])
plt.title(lossmon)
plt.xlabel("Dataset iterations")
vallossnamefile = "models/preprocessing_backgroundremoval/loss/model_Prep_BR_" + name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + \
"_loss_" + lossU + "_" + \
text_lr + "_" + lossmon+"_"+text_susc+"_datagenRecCircNewLoss"
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
