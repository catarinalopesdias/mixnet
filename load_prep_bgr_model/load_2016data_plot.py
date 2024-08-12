#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:24:17 2024

@author: catarinalopesdias

load 2016 data and plot
"""



folder_dir ="datareal/qsm2016_recon_challenge/data/"
#from datahandling import read_and_decode_tf
import nibabel as nib
import  matplotlib.pyplot as plt
###################
#load wrap data 
###################
#file = "phs_wrap.nii.gz"
#filetoupload = folder_dir + file
#img = nib.load(filetoupload)
#input_phasebgwrap = img.get_fdata()
#input_phasebgwrap = input_phasebgwrap[np.newaxis, :,:,:, np.newaxis]


###################
# load unwrap data
###################
file = "phs_unwrap.nii.gz"
filetoupload = folder_dir + file
img = nib.load(filetoupload)
input_phasebgunwrap = img.get_fdata()
#input_phasebgunwrap = input_phasebgunwrap[np.newaxis, :,:,:, np.newaxis]

###################
# load mask data
###################

#loas mask
file = "msk.nii.gz"
filetoupload = folder_dir + file
img = nib.load(filetoupload)
mask = img.get_fdata()
########################################


#######################
#phase unwrap
plt.imshow(input_phasebgunwrap[80,:,:], cmap='gray',  vmin=-4, vmax=4)   
plt.show()

plt.imshow(input_phasebgunwrap[:,80,:], cmap='gray',  vmin=-4, vmax=4)   
plt.show()

plt.imshow(input_phasebgunwrap[:,:,80], cmap='gray',  vmin=-4, vmax=4)   
plt.show()
#####################
##############
from scipy import ndimage
import numpy as np


######################################
#sagittal
#plt.imshow(input_phasebgunwrap[80,:,:], cmap='gray',  vmin=-5, vmax=5)   
#plt.show()
#plt.imshow(np.transpose(input_phasebgunwrap[80,:,:]), cmap='gray',  vmin=-5, vmax=5)   
#plt.show()
rotated_img = ndimage.rotate(np.transpose(input_phasebgunwrap[80,:,:]), 180)
plt.imshow(rotated_img, cmap='gray',  vmin=-4, vmax=4)   
plt.show()

################
#coronal
#plt.imshow(input_phasebgunwrap[:,80,:], cmap='gray',  vmin=-4, vmax=4)   
#plt.show()
rotated_img = ndimage.rotate(input_phasebgunwrap[:,80,:], 90) # rotate 90 degrees left
plt.imshow(rotated_img, cmap='gray',  vmin=-5, vmax=5) 
plt.show()
##########################################################

# axial
#plt.imshow(input_phasebgunwrap[:,:,80], cmap='gray',  vmin=-4, vmax=4)   
#plt.show()
rotated_img = ndimage.rotate(input_phasebgunwrap[:,:,80], 90) # rotate 90 degrees left
plt.imshow(rotated_img, cmap='gray',  vmin=-5, vmax=5) 
plt.show()





