
"""
creates susceptibility 3d matrix"""

from keras.layer import Lambda
from keras import backend as K
import tensorflow as tf
size = 128  # [128,128,128]
#rect_num = 200

def simulate_susceptibility_sources_uni():
      
   simulation_dim = 128, 
   rectangles_total = 200,#800
   #pheres_total = 80,
   shape_size_min_factor = 0.01,
   shape_size_max_factor = 0.5
  
    #3d matrix with zeros -size sim dim
   temp_sources = tf.zeros((simulation_dim, simulation_dim, simulation_dim))
  
   shrink_factor_all = []
   susceptibility_all = []
   shape_size_all = tf.zeros((2,rectangles_total))

   #shapes=0,..., rect total -1
   for shapes in range(rectangles_total):

      # From 1 to almost 0.5       
      shrink_factor = 1/(   (shapes/rectangles_total + 1))
      
      shrink_factor_all.append(shrink_factor)
      
      shape_size_min = tf.floor(simulation_dim * shrink_factor * shape_size_min_factor)
      shape_size_max = tf.floor(simulation_dim * shrink_factor * shape_size_max_factor)
      

      
      shape_size_all[0,shapes] = shape_size_min
      shape_size_all[1,shapes] = shape_size_max

      ####
      susceptibility_value = tf.random.uniform(low=-0.2, high=0.2)
      
      #size of cuboid - random within siye min and max
      random_sizex = tf.random.randint(low=shape_size_min, high=shape_size_max)
      random_sizey = tf.random.randint(low=shape_size_min, high=shape_size_max)
      random_sizez = tf.random.randint(low=shape_size_min, high=shape_size_max)
      
      #position of cuboid (random inside the cube)
      x_pos = tf.random.randint(simulation_dim)
      y_pos = tf.random.randint(simulation_dim)
      z_pos = tf.random.randint(simulation_dim)

      # make sure it does not get out of the cube
      x_pos_max = x_pos + random_sizex
      if x_pos_max >= simulation_dim:
          x_pos_max = simulation_dim

      y_pos_max = y_pos + random_sizey
      if y_pos_max >= simulation_dim:
          y_pos_max = simulation_dim

      z_pos_max = z_pos + random_sizez
      if z_pos_max >= simulation_dim:
          z_pos_max = simulation_dim

      # change the sus values in the cuboids  
      temp_sources[x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max] = susceptibility_value
      susceptibility_all.append(susceptibility_value)


def simulate_susceptibility_sources_uni_layer():
  return simulate_susceptibility_sources_uni()


lambda_output= Lambda(simulate_susceptibility_sources_uni_layer)()

