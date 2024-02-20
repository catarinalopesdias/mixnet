
import tensorflow as tf
import matplotlib.pyplot as plt
from datahandling import read_and_decode_tf
import os
import numpy as np
import scipy.ndimage


##############################################################
### Preprocess data for testing
#################################################################
tfrecord_files = []


tfrecord_dir = "/home/catarinalopesdias/proj_6again/deepQSM/data/processed/trn" 

# List all files in the directory
files_in_directory = os.listdir(tfrecord_dir)

for file in files_in_directory:
    if file.endswith(".tfrecords"):
        full_path =os.path.join(tfrecord_dir, file)
        tfrecord_files.append(full_path)

tfrecord_dataset = tf.data.TFRecordDataset(tfrecord_files)

# Apply the parsing function to each record
parsed_dataset = tfrecord_dataset.map(read_and_decode_tf)




def plot_tf_files(image_3dI,image_3dO, slice_nbr, vmin, vmax,title):
#   print('Matrix size: {}'.format(image_3d.shape))
  fig = plt.figure(figsize=(10, 6))
  
  plt.suptitle(title, fontsize=10)
  (subfig1, subfig2) = fig.subfigures(2, 1) # create 2x1 subfigures

  ax1 = subfig1.subplots(1, 3)       # create 1x2 subplots on subfig1
  ax2 = subfig2.subplots(1, 3)       # create 1x2 subplots on subfig2
  subfig1.suptitle('Input')               # set suptitle for subfig1

  #plt.subplot(231)
  ax1[0].imshow(np.take(image_3dI, slice_nbr, 2), vmin=vmin, vmax=vmax, cmap='gray')
  ax1[0].set_title('Axial')

  #plt.subplot(232)
  image_rot = scipy.ndimage.rotate(np.take(image_3dI, slice_nbr, 1),90)
  ax1[1].imshow(image_rot, vmin=vmin, vmax=vmax, cmap='gray')
  ax1[1].set_title('Coronal')

  #plt.subplot(233)
  image_rot = scipy.ndimage.rotate(np.take(image_3dI, slice_nbr, 0),90)
  im = ax1[2].imshow(image_rot, vmin=vmin, vmax=vmax, cmap='gray')
  ax1[2].set_title('Sagittal');
  
  ########
  cbar = subfig1.colorbar(im, ax=ax1.ravel().tolist(), shrink=0.95)
  cbar.set_ticks(np.arange(vmin,vmax))


  ##########################################################
  ##########################################################
  ##########################################################

  #plt.suptitle(title2, fontsize=10)
  subfig2.suptitle('Output')               # set suptitle for subfig1

  #plt.subplot(234)
  ax2[0].imshow(np.take(image_3dO, slice_nbr, 2), vmin=vmin, vmax=vmax, cmap='gray')
  ax2[0].set_title('Axial');
  

  #plt.subplot(235)
  image_rot = scipy.ndimage.rotate(np.take(image_3dO, slice_nbr, 1),90)
  ax2[1].imshow(image_rot, vmin=vmin, vmax=vmax, cmap='gray')
  ax2[1].set_title('Coronal');

  #plt.subplot(236)
  image_rot = scipy.ndimage.rotate(np.take(image_3dO, slice_nbr, 0),90)
  im = ax2[2].imshow(image_rot, vmin=vmin, vmax=vmax, cmap='gray')
  ax2[2].set_title('Sagittal');
  
  cbar2 = subfig2.colorbar(im, ax=ax2.ravel().tolist(), shrink=0.95)
  cbar2.set_ticks(np.arange(vmin,vmax))



##########################################################################
dataset_np = parsed_dataset.as_numpy_iterator()

counter = 0
for data_sample in dataset_np:
    input_ds, output_ds = data_sample
    counter = counter+1
    tit_i ="input "+str(counter)
    tit_o ="output "+str(counter)
    if counter == 1:
        plot_tf_files(input_ds,output_ds,40, -0.5, 0.6,str(counter))
