#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:49:24 2024

@author: catarinalopesdias
"""
from keras.layer import Lambda
from keras import backend as K
import tensorflow as tf

def generate_3d_dipole_kernel(data_shape, voxel_size, b_vec):
    fov = tf.array(data_shape) * tf.array(voxel_size)

    ry, rx, rz = tf.meshgrid(tf.arange(-data_shape[1] // 2, data_shape[1] // 2),
                             tf.arange(-data_shape[0] // 2, data_shape[0] // 2),
                             tf.arange(-data_shape[2] // 2, data_shape[2] // 2))

    rx, ry, rz = rx / fov[0], ry / fov[1], rz / fov[2]

    sq_dist = rx ** 2 + ry ** 2 + rz ** 2
    sq_dist[sq_dist == 0] = 1e-6
    d2 = ((b_vec[0] * rx + b_vec[1] * ry + b_vec[2] * rz) ** 2) / sq_dist
    kernel = (1 / 3 - d2)

    return kernel


def forward_convolution_with_dipole(chi_sample):
    
    scaling = tf.sqrt(chi_sample.size)
    chi_fft = tf.fft.fftshift(tf.fft.fftn(tf.fft.fftshift(chi_sample))) / scaling
    
    chi_fft_t_kernel = chi_fft * generate_3d_dipole_kernel(chi_sample.shape, voxel_size=1, b_vec=[0, 0, 1])
   
    tissue_phase = tf.fft.fftshift(tf.fft.ifftn(tf.fft.fftshift(chi_fft_t_kernel)))
    tissue_phase = tf.real(tissue_phase * scaling)

    return tissue_phase
 
lambda_output= Lambda(forward_convolution_with_dipole)(input)
 

    
"""
    def calc_dipole_kernel(dim):
        Calculates the dipole kernel of dimension 'dim'
        (1/3 - k_z^2/k^2)

        Parameters
        ----------
        dim: list
            A list of sampling positions in each dimension

        Returns
        -------
        tf tensor
            Dipole kernel of dimension dim
        

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
        Calculates the forward QSM simulation, i.e. apply
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
        

        shape = data.get_shape()

        # TODO: padding size should actually be larger
        padding_size = int(shape.as_list()[0] / 8)

        data = tf.pad(
            data, [[padding_size, padding_size], [padding_size, padding_size],
                   [padding_size, padding_size]], "CONSTANT")

        ##############################

        f_kernel = self.calc_dipole_kernel(data.get_shape().as_list())

        # inverse fft to get the final result
        data_real = data
        data_imag = tf.zeros_like(data)  # tensorflow only supports complex fft
        data = tf.complex(data_real, data_imag)

        f_kernel_real = f_kernel
        f_kernel_imag = tf.zeros_like(f_kernel)
        f_kernel = tf.complex(f_kernel_real, f_kernel_imag)

        f_data = common.fftshift(tf.signal.fft3d(data))#changed
        f_result = tf.multiply(f_kernel, f_data)
        result = tf.signal.ifft3d(common.ifftshift(f_result)) #changed

        #############################

        start = tf.constant([padding_size] * 3)
        result = tf.slice(tf.math.real(result), start, shape)
        f_kernel = tf.slice(tf.math.real(f_kernel), start, shape)
        f_data = tf.slice(tf.math.real(f_data), start, shape)
        f_result = tf.slice(tf.math.real(f_result), start, shape)

        return result, f_kernel, f_data, f_result

    def forward_simulation_full_tf(self, data):
        Applies the forward simulation (with full output) by running a
        tensorflow session.

        Parameters
        ----------
        data : numpy array
            Input data

        Returns
        -------
        numpy array
            Result
        

        graph = tf.Graph()

        with graph.as_default():

            # to remove the tensorflow log messages
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # input values
            tf_input_data = tf.compat.v1.placeholder(tf.float32, shape=data.shape) #changed

            # add backgroundfield calculation to the default graph
            tf_result, tf_f_kernel, tf_f_data, tf_f_result = self.forward_simulation(
                tf_input_data)
            #tf_output_data, *_ = self.forward_simulation(tf_input_data)

            self.logger.debug("input NAME: {}".format(tf_input_data.name))
            self.logger.debug("output NAME: {}".format(tf_result.name))

            # calculate backgroundfield
            sess = tf.compat.v1.Session()

            start_time = time.time()

            result, f_kernel, f_data, f_result = sess.run(
                (tf_result, tf_f_kernel, tf_f_data, tf_f_result),
                feed_dict={tf_input_data: data})

            duration = time.time() - start_time
            self.logger.debug(
                "forward simulation: {:.3f} sec".format(duration))

            sess.close()
            sess = None
            # to set log messages back to default
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            del graph
            graph = None

        return result, f_kernel, f_data, f_result

    def forward_simulation_tf(self, data, graph=None, sess=None, kill=True):
        Applies the forward simulation by running a tensorflow session.

        Parameters
        ----------
        data : numpy array
            Input data
        graph : tf graph
            graph that should be used, if none a new graph is created
        sess : tf sess
            tenorflow session
        kill : boolean
            indicates if we should destroy the session and the graph

        Returns
        -------
        numpy array
            Result
        tf graph
            graph
        tf session
            session
        

        create_graph = False
        if graph is None:
            graph = tf.Graph()
            create_graph = True

        with graph.as_default():

            if create_graph:
                # to remove the tensorflow log messages
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

                # input values
                tf_input_data = tf.compat.v1.placeholder(tf.float32, shape=data.shape)

                # add backgroundfield calculation to the default graph
                tf_output_data, *_ = self.forward_simulation(tf_input_data)

                self.logger.debug("input NAME: {}".format(tf_input_data.name))
                self.logger.debug("output NAME: {}".format(
                    tf_output_data.name))

                # calculate backgroundfield
                sess = tf.compat.v1.Session()

            else:
                tf_output_data = graph.get_tensor_by_name("Slice:0")
                tf_input_data = graph.get_tensor_by_name("Placeholder:0")

            start_time = time.time()

            data = sess.run(tf_output_data, feed_dict={tf_input_data: data})

            duration = time.time() - start_time
            self.logger.debug(
                "forward simulation: {:.3f} sec".format(duration))

            if kill:
                sess.close()
                sess = None
                # to set log messages back to default
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
                del graph
                graph = None

        return data, graph, sess
"""
    