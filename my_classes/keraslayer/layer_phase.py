#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:55:37 2024

@author: catarinalopesdias
"""

from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
import numpy as np

from tensorflow.experimental import numpy as tnp # start using tnp instead of numpy or math library#test tnp.pi, tnp.e
#from create_datasetfunctions_susc_norm01 import forward_convolution
#from create_datasetfunctions import forward_convolution


class CreatePhaseLayer(Layer):
    
    def __init__(self):
        super().__init__()
        
    def ifftshift(self, tensor):
        """Calculates ifftshift for a given tf tensor.
    
        Parameters
        ----------
        tensor : tf-tensor
            Input data
    
        Retruns
        -------
        tf-tensor
            Tensor after ifftshift.
        """
        ndims = len(tensor.get_shape())
        for idx in range(ndims):
            left, right = tf.split(tensor, 2, idx)
            tensor = tf.concat([right, left], idx)
        return tensor    
    
    def fftshift(self, tensor):
       """ Calculates fftshift for a given tf tensor.
    
        Parameters
        ----------
        tensor : tf-tensor
            Input data
    
        Retruns
        -------
        tf-tensor
            Tensor after fftshift.
      """  
      
     
       ndims = len(tensor.get_shape())
       for idx in range(ndims - 1, -1, -1):
            left, right = tf.split(tensor, 2, idx)
            tensor = tf.concat([right, left], idx)
       return tensor
    
    

    
    def calc_dipole_kernel(self,dim):
        """Calculates the dipole kernel of dimension 'dim'
        (1/3 - k_z^2/k^2)

        Parameters
        ----------
        dim: list
            A list of sampling positions in each dimension

        Returns
        -------
        tf tensor
            Dipole kernel of dimension dim
        """

        linspace = [
            tf.linspace(0.0, float(dim[i] - 1), dim[i])
            for i in range(len(dim))
        ]
        coord = tf.meshgrid(*linspace, indexing='ij')

        # move origin to the center
        coord = [coord[i] - (dim[i] - 1.0) / 2.0 for i in range(len(dim))]

        # calculate squared magnitude
        magnitude = [tf.square(coord[i]) for i in range(len(dim))]
        magnitude = sum(magnitude)
        #magnitude = tf.sqrt(magnitude)

        kernel = 1.0 / 3.0 - tf.square(coord[2]) / magnitude

        return kernel

    def forward_simulation(self, data):
        """Calculates the forward QSM simulation, i.e. apply
        dipole kernel in the fourier domain.

        Parameters
        ----------
        data : tf tensor
            Input data

        Returns
        -------
        tf tensor
            Result
        tf tensor
            Kernel in fourier domain
        tf tensor
            Data in fourier domain
        tf tensor
            Result in fourier domain
        """

        shape = [128,128,128]#data.get_shape()

        # TODO: padding size should actually be larger
        padding_size = int(128/8)#int(shape.as_list()[0] / 8)

        data = tf.pad(
            data, [[padding_size, padding_size], [padding_size, padding_size],
                   [padding_size, padding_size]], "CONSTANT")

        ##############################

        #f_kernel = self.calc_dipole_kernel(data.get_shape().as_list())
        f_kernel = self.calc_dipole_kernel([160,160,160])



        # inverse fft to get the final result
        data_real = data
        data_imag = tf.zeros_like(data)  # tensorflow only supports complex fft
        data = tf.complex(data_real, data_imag)

        f_kernel_real = f_kernel
        f_kernel_imag = tf.zeros_like(f_kernel)
        f_kernel = tf.complex(f_kernel_real, f_kernel_imag)

        f_data = self.fftshift(tf.signal.fft3d(data))#changed
        f_result = tf.multiply(f_kernel, f_data)
        result = tf.signal.ifft3d(self.ifftshift(f_result)) #changed

        #############################

        start = tf.constant([padding_size] * 3)
        result = tf.slice(tf.math.real(result), start, shape)
        f_kernel = tf.slice(tf.math.real(f_kernel), start, shape)
        f_data = tf.slice(tf.math.real(f_data), start, shape)
        f_result = tf.slice(tf.math.real(f_result), start, shape)

        return result, f_kernel, f_data, f_result
        
    def call(self, input_tensor):# inputs
        print("----start phase---")
        data = input_tensor
        print("data shape", data.shape)
        data =  data[0,:,:,:,0] # IF 4 D  

        sim_fwgt,ffw,ffw,ffd = self.forward_simulation(data)

        sim_fwgt = tf.expand_dims(sim_fwgt, 0)
        sim_fwgt = tf.expand_dims(sim_fwgt, 4)
       #return output_data    
        return sim_fwgt#sim_fwgt
 

#####################################################################################
"""import numpy as np
from plotting.visualize_volumes import view_slices_3dNew
from create_datasetfunctions_susc_unif02 import simulate_susceptibility_sources_uni
from plotting.visualize_volumes import view_slices_3dNew
import tensorflow as tf
size = 128  
rect_num = 200


#gt
gt = simulate_susceptibility_sources_uni( #simulate_susceptibility_sources_uni
    simulation_dim=size, rectangles_total=rect_num, plot=False)

gt = np.expand_dims(gt, 0)
gt = np.expand_dims(gt, 4)

#convert to tensorflow
gt = tf.convert_to_tensor(gt)


## input for layer - phase with mask, mask

#view_slices_3dNew( gt[0,:,:,:,0], 50, 50, 50,
#                  vmin=-0.5, vmax=0.5, title="gt")

inputs = [gt]


LayerConv= CreatePhaseLayer()

output = LayerConv(inputs)

view_slices_3dNew( gt[0,:,:,:,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="phase with background field and mask")

view_slices_3dNew( output[0,:,:,:,0], 50, 50, 50,
                  vmin=-0.5, vmax=0.5, title="phase with background field and mask")

"""

