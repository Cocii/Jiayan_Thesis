"""

Here we define the U-net architecture for super-resolution of 2D images
We insert a Reflection Padding layer here!
Reference code: https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb

"""
import keras
import keras.callbacks as cb
from keras import Input, Model
from keras.layers import Convolution2D, Activation, BatchNormalization, Conv2DTranspose, UpSampling2D, MaxPooling2D, Concatenate
import tensorflow as tf
from tensorflow import pad
# from keras.engine.until import Layer
# from keras.engine import InputSpec
from tensorflow.keras.layers import Layer, InputSpec

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')
        
    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        print(config)
        return config

def down_block(x, filters, kernel_size=(3, 3), padding="valid", strides=1):
    p = ReflectionPadding2D(padding=(1,1))(x)
    c = Convolution2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(p)
    c = BatchNormalization(axis=1)(c)
    p = ReflectionPadding2D(padding=(1,1))(c)
    c = Convolution2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(p)
    c = BatchNormalization(axis=1)(c)
    p = MaxPooling2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="valid", strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    p = ReflectionPadding2D(padding=(1,1))(concat)
    c = Convolution2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(p)
    c = BatchNormalization(axis=1)(c)
    p = ReflectionPadding2D(padding=(1,1))(c)
    c = Convolution2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(p)
    c = BatchNormalization(axis=1)(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    #p = ReflectionPadding2D(padding=(1,1))(x)
    c = Convolution2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = BatchNormalization(axis=1)(c)
    #p = ReflectionPadding2D(padding=(1,1))(c)
    c = Convolution2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = BatchNormalization(axis=1)(c)
    return c


def uNet1(first_dim,second_dim):
    print("=====================first_dim:",first_dim,"=====================second_dim:",second_dim)
    f = [16, 32, 64, 128, 256]
    inputs = input_img = Input((first_dim, second_dim, 1))  #le due dimensioni devono essere potenze di 2!!  the two dimensions must be powers of 2!!!
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    c4, p4 = down_block(p3, f[3])
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3])
    u2 = up_block(u1, c3, f[2])
    u3 = up_block(u2, c2, f[1])
    u4 = up_block(u3, c1, f[0])
    
    outputs = Convolution2D(1, (1, 1), padding="same", activation="sigmoid")(u4)

    model = Model(inputs, outputs)
    
    return model

'''uNet = uNet1(32,1024)
uNet.summary()'''