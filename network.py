#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:23:10 2023
This is one U net network
@author: catarinalopesdias
"""
import os
import tensorflow as tf


from keras.layers import Input ,Conv3D, Conv3DTranspose, LeakyReLU, UpSampling3D, Concatenate , Add, BatchNormalization
from keras.models import Model
from keras.initializers import GlorotNormal

def build_CNN(input_tensor):


#########################################################################################################################
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(input_tensor)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc3 = Add()([X_save, X])

    #down convolutional layer
    encoding_down_1 = Conv3D(filters=16,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2))(X_conc3)
    batch_norm_layer_3 = BatchNormalization()(encoding_down_1)
    # batch_norm_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_3)                    #i dont know if i need that-check it

#############################################################################################################################
    #Second Layer-->down
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc2 = Add()([X_save, X])


    encoding_down_2 = Conv3D(filters=32,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2)
                            )(X_conc2)
    batch_norm_layer_2 = BatchNormalization()(encoding_down_2)
    # batch_norm_layer_2 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_2)
###################################################################################################################################################
    #Third Layer-->down
    # First convolutional layer with activation and batch normalization
    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_2)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    # Second convolutional layer with activation and batch normalization
    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    # Add the input tensor to the output tensor (residual connection)
    X_conc1 = Add()([X_save, X])
    encoding_down_3 = Conv3D(filters=64,
                            kernel_size=[3, 3, 3],
                            activation=LeakyReLU(alpha=0.2),
                            padding='same',
                            kernel_initializer=GlorotNormal(42),
                            strides=(2,2,2)
                            )(X_conc1)

    batch_norm_layer_3 = BatchNormalization()(encoding_down_3)
    # batch_norm_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(batch_norm_layer_3)
#######################################################################################################################
    #Fourth Layer = Connection between Layer
    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(batch_norm_layer_3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=128, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

#####################################################################################################
    #Third Layer --> up NOte: if not workin, you have to slice
    decoder_up_1 = UpSampling3D(size=(2,2,2))(X)
    decoder_1 = Conv3DTranspose(filters=64,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_1)

    #vertical connection between these convolutional layers--> adds the output together
    #TensorShape([None, 3, 32, 32, 64])


    combine_conc1_dec1 = Concatenate()([X_conc1, decoder_1])

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(combine_conc1_dec1)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=64, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
##########################################################################################################
    #Second layer -->up
    decoder_up_2 = UpSampling3D(size=(2,2,2))(X)
    decoder_2 = Conv3DTranspose(filters=32,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_2)

    #vertical connection between these convolutional layers--> adds the output together

    combine_conc2_dec2 = Concatenate()([X_conc2, decoder_2])

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(combine_conc2_dec2)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=32, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
########################################################################################################################
    # Third Layer --> up
    decoder_up_3 = UpSampling3D(size=(2,2,2))(X)
    decoder_3 = Conv3DTranspose(filters=16,
                        kernel_size=[3,3,3],
                        activation=LeakyReLU(alpha=0.2),
                        padding='same')(decoder_up_3)

    combine_conc3_dec3 = Concatenate()([X_conc3, decoder_3])

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(combine_conc3_dec3)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X_save = X

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)

    X = Conv3D(filters=16, kernel_size=[3, 3, 3],padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    X = BatchNormalization()(X)
###################################################################################################################
    # output before residual conenction
    output_layer = Conv3D(filters=1 ,kernel_size=[3,3,3], padding='same')(X)

    #residual connection between input and output
    residual_conn = Add()([input_tensor, output_layer])
    output_tensor=residual_conn

    #r#eturn output_tensor
    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
    #return tf.keras.Model(inputs=input_tensor, outputs=output_tensor), output_tensor