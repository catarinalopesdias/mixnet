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
from plotting.visualize_volumes import view_slices_3dNew

################################################
#   Import data
################################################
#view input
#view output

#phase_bg = np.load('syntheticdata/phase_bg100.npy')
#phase = np.load('syntheticdata/phase100.npy')
######
# compressed data
#loaded = np.load('datasynthetic/150samples.npz')
loaded = np.load('datasynthetic/150backgroundfields_1gt_lessbg.npz')

phase = loaded['phase1']
phase_bg = loaded['phase_bg1']
del loaded
#######################


for num_slice in range(5):

   # view_slices_3dNew(phase[num_slice,:,:,:], 50, 50,50, vmin=-2, vmax=2, title="phase") 
    view_slices_3dNew(phase_bg[num_slice,:,:,:], 50, 50,50, vmin=-2, vmax=2, title="phase+bg")
