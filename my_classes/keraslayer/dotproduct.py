from keras import backend as K
#from keras.engine.topology import Layer
from keras.layers import Layer
import tensorflow as tf

#from tensorflow.keras.layers import Layer
#import numpy as np



#https://gist.github.com/nairouz/5b65c35728d8fb8ec4206cbd4cbf9bea
#https://www.tutorialspoint.com/keras/keras_customized_layer.htm
class MyCustomLayerDot(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim  = output_dim #sets output dim
        super(MyCustomLayerDot, self).__init__(**kwargs) # calls the base or super layer’s init function.


    def build(self, input_shape): # defines the build method with one argument
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=False) #  creates the weight corresponding to input shape and set it in the kernel. It is our custom functionality of the layer. It creates the weight using ‘normal’ initializer.
        super(MyCustomLayerDot, self).build(input_shape) # calls the base class, build method.

    # For our fully connected layer it means that we have to calculate the dot product between the weights and the input
    def call(self, x):
        y = K.dot(x, self.kernel)
        return y
    

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    #CHECK https://stackoverflow.com/questions/77785265/how-to-write-a-new-custom-layer-in-keras
    # https://stackoverflow.com/questions/58192501/how-to-multiply-a-layer-by-a-constant-vector-element-wise-in-keras