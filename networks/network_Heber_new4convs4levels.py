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


def build_CNN_Heber_inputoutput4convs4levels( input_tensor, filterN =16):
    
    filter_size = [3,3,3]
    num_filter = filterN
    bias_init = 1e-8
    seed = 66420
    
    #####################################
    X0_save = input_tensor
    ########################################################
    ###########################################################
    ##X1 
    # First convolutional layer with activation -----------------
    X = Conv3D(filters = num_filter, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(input_tensor)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)

    X1_save = X 
    
    # Second convolutional layer with activation  -------------------------
    X = Conv3D(filters = num_filter, 
             kernel_size = filter_size,
             activation=None,
             kernel_initializer=GlorotNormal(seed),
             bias_initializer=tf.constant_initializer(bias_init),
             padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolutional layer with activation ----------------
    X = Conv3D(filters = num_filter, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth convolutional layer with activation ----------------
    X = Conv3D(filters = num_filter, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    # (None, 128, 128, 128, 16) 
    
    # residual connection -------------------------------------
    X1_conc =  Average()([X1_save, X])
    
    #(None, 128, 128, 128, 16) 
    #####################################################################
    #####################################################################
    # X1down  
    #down convolutional layer 
    X1_down = Conv3D(filters = num_filter,
                            kernel_size = filter_size,
                            #activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(seed),
                            bias_initializer=tf.constant_initializer(bias_init),
                            strides=(2,2,2))(X1_conc)
    
    X1_down = LeakyReLU(alpha=0.2)(X1_down)
    #(None, 64, 64, 64, 16) 
    ##################################################################
    ##################################################################
    # X2 - second layer --> down
    
    # First convolutional layer with activation --------------------
    X = Conv3D(filters = num_filter * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X1_down)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    # (None, 64, 64, 64, 32) 
    X2_save = X 
    
    # Second convolutional layer with activation ----------
    X = Conv3D(filters= num_filter * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolutional layer with activation  ----------
    X = Conv3D(filters = num_filter * 2, #32
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth convolutional layer with activation  ----------
    X = Conv3D(filters = num_filter * 2, #32
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)

    # residual connection
    X2_conc =  Average()([X2_save, X])
    #(None, 64, 64, 64, 32)  
    
    ################################################
    ##########################################################
    #X2_down

    X2_down = Conv3D(filters = num_filter * 2,
                        kernel_size=filter_size,
                        #activation=LeakyReLU(alpha=0.2),
                        padding='same',
                        kernel_initializer=GlorotNormal(seed),
                        strides=(2,2,2)
                        )(X2_conc)
    
    X2_down = LeakyReLU(alpha=0.2)(X2_down)
    
    #(None, 32, 32, 32, 32)   
    ###################################################
    ###################################################
    #X3 -third layer

    # First convolutional layer with activation ----------
    X = Conv3D(filters = num_filter *2 * 2,
               kernel_size=filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
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
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    #########
    #third convolutional layer with activation ----------
    X = Conv3D(filters = num_filter * 2 * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    #fourth convolutional layer with activation ----------
    X = Conv3D(filters = num_filter * 2 * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    
    #residual connection ---------------------------------
    X3_conc =  Average()([X3_save, X])

    
    #(None, 32, 32, 32, 64)
    ##################################################
    ##################################################

    #X3down

    X3_down = Conv3D(filters = num_filter * 2 * 2,
                        kernel_size=filter_size,
                        #activation=LeakyReLU(alpha=0.2),
                        padding='same',
                        kernel_initializer=GlorotNormal(seed),
                        strides=(2,2,2)
                        )(X3_conc)
    
    X3_down = LeakyReLU(alpha=0.2)(X3_down)
    
    #(None, 16, 16, 16, 32)   
    ##################################################
    ##################################################

    #X4 -fourth layer

    # First convolutional layer with activation ----------
    X = Conv3D(filters = num_filter *2 * 2 * 2,
               kernel_size=filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X3_down)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    #########
    X4_save = X 
    #######
    
    # Second convolutional layer with activation ----------
    X = Conv3D(filters = num_filter * 2 * 2 * 2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    #########
    #third convolutional layer with activation ----------
    X = Conv3D(filters = num_filter * 2 * 2 * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    #fourth convolutional layer with activation ----------
    X = Conv3D(filters = num_filter * 2 * 2 * 2,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    
    #residual connection ---------------------------------
    X4_conc =  Average()([X4_save, X])

    #############################################
    ############################################

    X4_up = UpSampling3D(size=(2,2,2))(X4_conc)
    
    X4_up = Conv3D(filters = num_filter * 2 * 2, 
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(X4_up)


#########################################################
#########################################################
#########################################################
###########################################################
    #X5
    combine_conc_x3x4up  = Concatenate()([X3_conc, X4_up])


    #----------- level ----------------------------------------------------
    #first convolution -------------------------------
    X = Conv3D(filters = num_filter * 2 * 2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(combine_conc_x3x4up)
    
    X = LeakyReLU(alpha=0.2)(X)

    X5_save = X

    #second convolution ------------------------------------------------
    X = Conv3D(filters = num_filter * 2 * 2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolution ------------------------------------
    X = Conv3D(filters = num_filter * 2 * 2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth  convolution-------------------------------------
    X = Conv3D(filters = num_filter * 2 * 2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # residual connection ------------------------------------------

    X5_conc =  Average()([X5_save, X])
    #return X4_conc 

    ##############################################################################


    #X5up

    X5_up = UpSampling3D(size=(2,2,2))(X5_conc)
    
    
    X5_up = Conv3D(filters = num_filter * 2, 
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(X5_up)
    
    #vertical connection between these convolutional layers--> adds the output together
    #TensorShape([None, 3, 32, 32, 64])
    

    ###########################################################
    #X6

    combine_conc_x2x5up  = Concatenate()([X2_conc, X5_up])


    #----------- level ----------------------------------------------------
    #first convolution -------------------------------
    X = Conv3D(filters = num_filter *2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(combine_conc_x2x5up)
    
    X = LeakyReLU(alpha=0.2)(X)

    X6_save = X

    #second convolution ------------------------------------------------
    X = Conv3D(filters = num_filter*2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolution ------------------------------------
    X = Conv3D(filters = num_filter*2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth  convolution-------------------------------------
    X = Conv3D(filters = num_filter*2, 
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # residual connection ------------------------------------------
    X6_conc =  Average()([X6_save, X])
    #return X6_conc 

########################################    
    #x6 up
    X6_up = UpSampling3D(size=(2,2,2))(X6_conc)
    
    X6_up = Conv3D(filters = num_filter,
                            kernel_size = filter_size,
                            activation=LeakyReLU(alpha=0.2),
                            padding='same')(X6_up)
############################################################################
    #X7

    combine_conc1_x6_up = Concatenate()([X1_conc, X6_up])


    #----------- level ----------------------------------------------------
    #first convolution -------------------------------
    X = Conv3D(filters = num_filter,
               kernel_size=filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(combine_conc1_x6_up)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)

    X7_save = X

    # second convolution -------------------------------
    X = Conv3D(filters = num_filter,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # third convolution -------------------------------
    X = Conv3D(filters = num_filter,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)
    
    # fourth convolution ----------------------------
    X = Conv3D(filters = num_filter,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init),
               padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    # = BatchNormalization()(X)
    
    # residual connection -----------------------------
    X7_conc =  Average()([X7_save, X])

    #########################

    ################
    #X8
    
    X = Conv3D(filters=1,
               kernel_size = filter_size,
               activation=None,
               kernel_initializer=GlorotNormal(seed),
               bias_initializer=tf.constant_initializer(bias_init), #tf.keras.initializers.Constant(bias_init)
               padding='same')(X7_conc)
    X = LeakyReLU(alpha=0.2)(X)
    #X = BatchNormalization()(X)


    ################################################
    ################################################
    
    output_tensor =  Average()([X0_save, X])

    
    return output_tensor
    ################################################
    ################################################
    



input_shape = (128, 128, 128, 1) # shape of pict, last is the channel
input_tensor = Input(shape = input_shape, name="input")


##########################################


model = Model(input_tensor, build_CNN_Heber_inputoutput4convs4levels(input_tensor))




model.summary()
