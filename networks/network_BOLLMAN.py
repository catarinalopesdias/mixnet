#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:23:10 2023
This is one U net network
this is based on Steffens bollman
"""
import os
import tensorflow as tf


from keras.layers import Input ,Conv3D, Conv3DTranspose, LeakyReLU, UpSampling3D, Concatenate , Add, BatchNormalization
from keras.models import Model
from keras.initializers import GlorotNormal

#####################################################################################

def downsample(filters, kernel_size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv3D(filters, kernel_size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

#####################################################################################
#####################################################################################


def upsample(filters, kernel_size, apply_dropout=False, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result




def build_CNN_BOLLMAN(input_tensor):

  filter_base = 32
  kernel_size = 3
  OUTPUT_CHANNELS = 1

  down_stack = [
    downsample(filter_base, kernel_size, apply_batchnorm=False), # (bs, 32xxx 64 if filter base = 64)
    downsample(filter_base*2, kernel_size), # (bs, 16xxx, 128)
    downsample(filter_base*3, kernel_size), # (bs, 8xxx, 256)
    downsample(filter_base*4, kernel_size), # (bs, 4xxx, 512)
    downsample(filter_base*5, kernel_size), # (bs, 2xxx, 512)
  ]

  up_stack = [
    upsample(filter_base*5, kernel_size, apply_dropout=True), # (bs, 16, 16, 1024)
    upsample(filter_base*4, kernel_size, apply_dropout=True), # (bs, 32, 32, 512)
    upsample(filter_base*3, kernel_size), # (bs, 64, 64, 256)
    upsample(filter_base*2, kernel_size), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, kernel_size,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None,None,None,1])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)



#    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
#    #return tf.keras.Model(inputs=input_tensor, outputs=output_tensor), output_tensor