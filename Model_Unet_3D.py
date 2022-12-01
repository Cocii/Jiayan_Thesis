"""

Here we define the U-net architecture for super-resolution of 2D images
We work with 3D tensors!
Reference code: https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb

"""
from keras import Input, Model
from keras.layers import *
from keras.initializers import RandomNormal
import tensorflow as tf

def batchnorm():
    gamma_init = RandomNormal(1., 0.02)
    #gamma_init = RandomNormal(10., 0.2)
    return BatchNormalization(momentum=0.9, axis=1, epsilon=1.01e-5, gamma_initializer=gamma_init)

def conv3d(f, *a, **k):
    conv_init = RandomNormal(0, 0.02)
    return Conv3D(f, kernel_initializer=conv_init, *a, **k)


def uNet3(num_freqs):

    def down_block(x, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
        c = Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = BatchNormalization(axis=1)(c)
        c = Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        c = BatchNormalization(axis=1)(c)
        p = MaxPooling3D((2, 2, 2), (2, 2, 2))(c)
        return c, p

    def up_block(x, skip, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
        '''x = Conv3DTranspose(filters, kernel_size=4, strides=2)(x)
        x = Cropping3D(1)(x)'''
        us = UpSampling3D((2, 2, 2))(x)
        concat = Concatenate()([us, skip])
        c = Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
        c = BatchNormalization(axis=1)(c)
        c = Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        c = BatchNormalization(axis=1)(c)
        return c

    def bottleneck(x, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
        #p = ReflectionPadding2D(padding=(1,1))(x)
        c = Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
        c = BatchNormalization(axis=1)(c)
        #p = ReflectionPadding2D(padding=(1,1))(c)
        c = Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
        c = BatchNormalization(axis=1)(c)
        return c


    f = [32, 64, 128, 256, 512]
    inputs = Input(shape=(16, 64, num_freqs, 1))  #le dimensioni devono essere potenze intere di 2!!
    
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
    
    outputs = Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(u4)

    model = Model(inputs, outputs)
    
    return model

def uNet1():

    def down_block(x, filters, y_dim, kernel_size=4, padding="same", strides=2):
        if y_dim==16:
            use_batchnorm=False
        else:
            use_batchnorm=True

        x = conv3d(filters, kernel_size=kernel_size, padding=padding, 
                    strides=strides, name='conv_{0}'.format(y_dim),use_bias=(not (use_batchnorm and y_dim > 2)))(x)

        if y_dim>2:
            if y_dim<16:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            return x,x2
        else:
            return x

        '''c = Conv3D(filters, kernel_size=3, padding=padding, strides=1, activation="relu")(x)
        c = BatchNormalization(axis=1)(c)
        c = Conv3D(filters, kernel_size=3, padding=padding, strides=1, activation="relu")(c)
        c = BatchNormalization(axis=1)(c)
        p = MaxPooling3D((2, 2, 2), (2, 2, 2))(c)

        return c,p'''

    def up_block(x, skip, filters, y_dim, kernel_size=4, padding="same", strides=2):

        if y_dim==16:
            use_batchnorm=False
        else:
            use_batchnorm=True

        x = Activation("relu")(x)

        x = Conv3DTranspose(filters, kernel_size=kernel_size, strides=strides,
                            kernel_initializer=conv_init,
                            name='convt_{0}'.format(y_dim), use_bias=not use_batchnorm)(x)
        x = Cropping3D(1)(x)

        if y_dim<16:
            x = batchnorm()(x, training=1)
            x = Dropout(0.5)(x, training=1)
            x = Concatenate(axis=-1)([skip, x])
        return x

    f = [64, 128, 256, 512]
    s = [16, 8, 4, 2]

    conv_init = RandomNormal(0, 0.02)

    inputs = input_tens
    
    p0 = inputs
    c1,lr1 = down_block(inputs, f[0], s[0])
    bn1,lr2 = down_block(lr1, f[1], s[1])
    bn2,lr3 = down_block(lr2, f[2], s[2])
    c4 = down_block(lr3, f[3], s[3])

    u1 = up_block(c4, bn2, f[2], s[3])
    u2 = up_block(u1, bn1, f[1], s[2])
    u3 = up_block(u2, c1, f[0], s[1])
    u4 = up_block(u3, None, 1, s[0])
    
    outputs = u4
    #outputs = Conv3D(1, 1, padding="same", activation="sigmoid", name='final_conv')(u4)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def UNETGnoBN(isize, nc_in=1, nc_out=1, ngf=64, fixed_input_size=True):

    def conv3d(f, *a, **k):
        conv_init = RandomNormal(0, 0.02)
        return Conv3D(f, kernel_initializer=conv_init, *a, **k)

    def nan_func(x):
        x = tf.where(tf.math.is_nan(x), tf.ones_like(x) * 0, x)
        # if x is nan use 1 * NUMBER else use element in x
        return x

    def mask_func_nan(x):
        return x[1] + tf.cast(tf.math.is_nan(x[0]), tf.float32) * x[2]

    max_nf = 8 * ngf
    conv_init = RandomNormal(0, 0.02)

    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        assert s >= 2 and s % 2 == 0
        if nf_next is None:
            nf_next = min(nf_in * 2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv3d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                   padding="same", name='conv_{0}'.format(s))(x)
        if s > 2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s // 2, nf_next)
            x = Concatenate(axis=-1)([x, x2])
        x = Activation("relu")(x)
        x = Conv3DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer=conv_init,
                            name='convt_{0}'.format(s))(x)
        if s == ngf:
            x = Cropping3D(1, name='unet_out')(x)
        else:
            x = Cropping3D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <= 8:
            x = Dropout(0.5)(x, training=1)
        return x

    s = isize if fixed_input_size else None
    inputs = Input(shape=(s, s*4, s*32, nc_in))
    inputs_block = Lambda(nan_func, output_shape=(isize, isize*4, isize*32, 1))(inputs)
    # false BN
    x = block(inputs_block, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    #dec = Lambda(mask_func_nan, output_shape=(isize, isize*4, isize*32, 1))([inputs, inputs_block, x])

    model = Model(inputs=inputs, outputs=x)

    return model

'''uNet = uNet3(512)
uNet.summary()'''

