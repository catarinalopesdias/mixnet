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
#from networks.network import build_CNN
#from networks.network_BOLLMAN import build_CNN_BOLLMAN
from networks.network_adaptedfrom_BOLLMAN import build_CNN_BOLLMAN
#from visualize_volumes import view_slices_3dNew
from  my_classes.DataGeneratordipinv_susc02_bgrem import DataGeneratorUniform




num_train_instances = 500 
samples_dic = []


for i in range(num_train_instances):

    samples_dic.append( str(i) + 'samples' )
   

#######################
partition_factor = 0.8

partition = {'train': samples_dic[0: int(partition_factor * num_train_instances)] , 
             'validation':  samples_dic[ -int(( 1-partition_factor) * num_train_instances): num_train_instances]}


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

#ushape1 = build_CNN(input_tensor)
#ushape2 = build_CNN(ushape1)

print("compile model")

#model = Model(input_tensor, ushape2) #get from orig deepQSM algorithm
#model = build_CNN(input_tensor)
model = build_CNN_BOLLMAN(input_tensor)
name = "Bollmann"

###############################################
###############################################
print("Model with gradient accumulation")
gaaccumsteps = 10;
#learningrate
lr =0.0005
text_lr = str(lr).split(".")[1] #"default" #

model = GradientAccumulateModel(accum_steps=gaaccumsteps, inputs=model.input, outputs=model.output)

# we changed to mean absolute error
lossU = "mse" #"mean_absolute_error"#"mse"# "mean_absolute_error" #"mse"    #mse
#opt = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9 ,beta_2 =0.999, epsilon=1e-8, gradient_accumulation_steps=gaaccumsteps )


# from model managerloss

optimizerMINE = Adam(
              learning_rate=lr, #0.0003,
                beta_1=0.9,
                beta_2=0.999,
               epsilon=1e-8
               ) #gradient_accumulation_steps=10



model.compile(optimizer=optimizerMINE, loss= lossU) #mean_absolute_error
#model.compile(optimizer='adam', loss= lossU) #mean_absolute_error

model.summary()

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
batch_size = 1
num_filter = 16
lossmon = "val_loss"
###############
# {epoch:04d}     #"cp-{epoch:04d}"+

checkpoint_path = "checkpoints/bgremovalmodel/Bg_" + name + "_newadam"+ str(num_filter)+ \
    "_trainsamples" + str(num_train_instances) + \
        "_datasetiter" + str(dataset_iterations) + "_batchsize" + str(batch_size)+ \
        "_gaaccum" + str(gaaccumsteps) + "_loss_" + lossU + "_" + text_lr +"_"+ lossmon+"_" + text_susc + "_datagen.ckpt"

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
                    callbacks = [cp_callback,earlystop], # pass callback to training for saving the model80
                    workers=6)



                    
loss_historyGA = history.history['loss']


#with open('loss_historyGA.pickle', 'wb') as f:
#pickle.dump([loss_historyGA, dataset_iterations], f)
###################

    
if not os.path.exists("models/backgroundremovalBOLLMAN"):
    os.makedirs("models/backgroundremovalBOLLMAN")

    
model_name1 = "models/backgroundremovalBOLLMAN/model_BR_" + name + \
"_newadam_" + str(num_filter)+"filters_trainsamples" + str(num_train_instances) + \
"_datasetiter"+ str(dataset_iterations) + "_batchsize" + str(batch_size) + "_gaaccum" + str(gaaccumsteps) + \
"_loss_" + lossU + "_" + text_lr + "_"+ lossmon + "_"+text_susc+"_datagen.keras"



model.save(model_name1)



#################################
#plot loss

lossfile_extensionpng =".png"
lossfile_extensiontxt =".txt"

if not os.path.exists("models/backgroundremovalBOLLMAN/loss"): 
    os.makedirs("models/backgroundremovalBOLLMAN/loss") 

plt.figure(figsize=(6, 3))
plt.plot(loss_historyGA)
#plt.ylim([0, loss_historyGA[-1]*2])
plt.title("Loss")
plt.xlabel("Dataset iterations")
lossnamefile = "models/backgroundremovalBOLLMAN/loss/modelBR_"+ name + \
"_newadam" + str(num_filter)+"trainsamples" + str(num_train_instances) \
+ "_datasetiter"+ str(dataset_iterations) + "_batchsize"+ str(batch_size)+ \
"_gaaccum"+ str(gaaccumsteps) + "_loss_" + lossU + "_" + text_lr + \
"_" + lossmon+"_"+text_susc+"_datagen"
plt.savefig(lossnamefile + lossfile_extensionpng )

###############
# plot loss as txt

file = open(lossnamefile+lossfile_extensiontxt,'w')
for item in loss_historyGA:

	file.write(str(item)+"\n")
file.close()