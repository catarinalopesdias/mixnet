#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:49:04 2024

@author: catarinalopesdias
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:23:10 2023
This is one U net network
@author: catarinalopesdias
"""
import os
import tensorflow as tf


from keras.layers import Input ,Conv3D, Conv3DTranspose, LeakyReLU, UpSampling3D, Concatenate , Add, BatchNormalization, Average
from keras.models import Model
from keras.initializers import GlorotNormal


##############################################################


def build_CNN_catarina_inputoutput(input_tensor):
    
    filter_size = [3,3,3]
    num_filter = 16
    bias_init = 1e-8
    
    #####################################
    X0_save = input_tensor
    ########################################################
    ##X1 
    # First convolutional layer with activation -----------------
    X = Conv3D(filters = num_filter, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(input_tensor)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)

    X1_save = X 
    
    # Second convolutional layer with activation  -------------------------
    X = Conv3D(filters = num_filter, 
             kernel_size = filter_size,
             activation=None,
             kernel_initializer=GlorotNormal(42),
             bias_initializer=tf.constant_initializer(bias_init),
             padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolutional layer with activation ----------------
    X = Conv3D(filters = num_filter, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth convolutional layer with activation ----------------
    X = Conv3D(filters = num_filter, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    # (None, 128, 128, 128, 16) 
    
    # residual connection -------------------------------------
    X1_conc =  Average()([X1_save, X])
    
    #(None, 128, 128, 128, 16) 
    #####################################################################
    ##########################################################################
    # X1down  
    #down convolutional layer 
    encoding_down_1 = Conv3D(filters = num_filter,
                            kernel_size = filter_size,
                            #activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            bias_initializer=tf.constant_initializer(bias_init),
                            strides=(2,2,2))(X1_conc)
    
    encoding_down_1 = LeakyReLU(alpha=0.2)(encoding_down_1)
    #(None, 64, 64, 64, 16) 
    ##################################################################
    ###############################################################
    # X2 - second layer --> down
    
    # First convolutional layer with activation --------------------
    X = Conv3D(filters = num_filter * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(encoding_down_1)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    # (None, 64, 64, 64, 32) 
    X2_save = X 
    
    # Second convolutional layer with activation ----------
    X = Conv3D(filters= num_filter * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolutional layer with activation  ----------
    X = Conv3D(filters = num_filter * 2, #32
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth convolutional layer with activation  ----------
    X = Conv3D(filters = num_filter * 2, #32
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)

    # residual connection
    X2_conc =  Average()([X2_save, X])
    #(None, 64, 64, 64, 32)  
    
    ################################################
    #X2_down

    X2_down = Conv3D(filters = num_filter * 2,
                        kernel_size=filter_size,
                        #activation=LeakyReLU(alpha=0.2),
                        padding='same',
                        kernel_initializer=GlorotNormal(42),
                        strides=(2,2,2)
                        )(X2_conc)
    
    X2_down = LeakyReLU(alpha=0.2)(X2_down)
    
    #(None, 32, 32, 32, 32)   
    ###################################################
    #X3 -third layer

    # First convolutional layer with activation ----------
    X = Conv3D(filters = num_filter *2 * 2,
               kernel_size=filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X2_down)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    #########
    X3_save = X 
    #######
    
    # Second convolutional layer with activation ----------
    X = Conv3D(filters = num_filter *2 * 2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    #########
    #third convolutional layer with activation ----------
    X = Conv3D(filters = num_filter * 2 * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    #fourth convolutional layer with activation ----------
    X = Conv3D(filters = num_filter * 2 * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    
    #residual connection ---------------------------------
    X3_conc =  Average()([X3_save, X])

    
    #(None, 32, 32, 32, 64)
    ##################################################
    #X3up

    X3_up = UpSampling3D(size=(2,2,2))(X3_conc)
    
    
    #X3_up = Conv3DTranspose(filters = num_filter * 2, #32
    #                    kernel_size=[3,3,3],
    #                    activation=LeakyReLU(alpha=0.2),
    #                    padding='same')(X3_up)
    
    X3_up = Conv3D(filters = num_filter * 2, #32
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(X3_up)
    
    #vertical connection between these convolutional layers--> adds the output together
    #TensorShape([None, 3, 32, 32, 64])
    

    ###########################################################
    #X4
    #combine_conc2_dec3 = Concatenate()([X2_conc, X3_up])
    #combine_conc2_dec3 = Add()([X2_conc, X3_up])

    #X3_upsave = combine_conc2_dec3


    #X = Conv3D(filters = num_filter * 2,
    #           kernel_size = filter_size,
    #           padding='same')(X3_up)
    
    #X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)

    combine_conc_x2x3up  = Concatenate()([X2_conc, X3_up])
    #combine_con_x2x3up= Add()([X2_conc, X3_up])
    #combine_con_x2x3up =  Average()([X2_conc, X3_up])


    #----------- level ----------------------------------------------------
    #first convolution -------------------------------
    X = Conv3D(filters = num_filter*2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(combine_conc_x2x3up)
    
    X = LeakyReLU(alpha=0.2)(X)

    X4_save = X

    #second convolution ------------------------------------------------
    X = Conv3D(filters = num_filter*2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X4_save)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolution ------------------------------------
    X = Conv3D(filters = num_filter*2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth  convolution-------------------------------------
    X = Conv3D(filters = num_filter*2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # residual connection ------------------------------------------
    #X4_conc = Add()([X4_save, X])
    X4_conc =  Average()([X4_save, X])
    #return X4_conc 

########################################    
    #x4 up
    X4_up = UpSampling3D(size=(2,2,2))(X4_conc)
    
    #X4_up = Conv3DTranspose(filters = num_filter,
    #                        kernel_size = filter_size,
    #                        activation=LeakyReLU(alpha=0.2),
    #                        padding='same')(X4_up)
    
    X4_up = Conv3D(filters = num_filter,
                            kernel_size = filter_size,
                            activation=LeakyReLU(alpha=0.2),
                            padding='same')(X4_up)
############################################################################
    #X5
    #X = Conv3D(filters = num_filter,
    #           kernel_size = filter_size,
    #           adding='same')(X4_up)
    #X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)

    combine_conc1_dec4 = Concatenate()([X1_conc, X4_up])
   # combine_conc1_dec4 = Add()([X1_conc, X4_up])
    #combine_conc1_dec4 =  Average()([X1_conc, X4_up])


    #----------- level ----------------------------------------------------
    #first convolution -------------------------------
    X = Conv3D(filters = num_filter,
               kernel_size=filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(combine_conc1_dec4)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)

    X5_save = X

    # second convolution -------------------------------
    X = Conv3D(filters = num_filter,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolution -------------------------------
    X = Conv3D(filters = num_filter,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth convolution ----------------------------
    X = Conv3D(filters = num_filter,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    # = BatchNormalization()(X)
    
    # residual connection -----------------------------
    X5_conc =  Average()([X5_save, X])

    #########################

    ################
    #X6
    
    X = Conv3D(filters=1,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(42),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X5_conc)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)


    ################################################
    ################################################
    
    #output_tensor = Add()([X0_save, X])
    output_tensor =  Average()([X0_save, X])

    
    return output_tensor
    ################################################
    ################################################
    



#input_shape = (None, None, None, 1) # shape of pict, last is the channel
input_shape = (128, 128, 128, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")




#y = tf.keras.layers.Concatenate()([maskedPhase, build_CNN_BOLLMANinputoutput(phasewithbackgroundfield)])
#y = tf.keras.layers.Concatenate()([input_tensor, build_CNN_catarina_inputoutput(input_tensor)])

##########################################


model = Model(input_tensor, build_CNN_catarina_inputoutput(input_tensor))

#name = "PhaseBgf_Bgfrem_cat"


model.summary()
