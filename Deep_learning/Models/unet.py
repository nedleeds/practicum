import os
import numpy as np


import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'} ==> not for see the gpu contents
from tensorflow                  import keras
from tensorflow.keras            import Model, layers
from tensorflow.keras.layers     import Input, Layer, Conv2D, MaxPool2D, UpSampling2D, Dropout
from tensorflow.keras.layers     import Conv2DTranspose, concatenate, Activation, BatchNormalization
from keras.layers.convolutional  import Convolution2D
from tensorflow.keras.optimizers import Adam
from keras import backend as K

# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

class MaxPoolingWithArgmax2D(Layer):
    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "padding" : self.padding,
            "pool_size" : self.pool_size,
            "strides" : self.strides
            })
        return config

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, *pool_size, 1]
        padding = padding.upper()
        strides = [1, *strides, 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding)

        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3])

        ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                            K.flatten(updates),
                            [K.prod(output_shape)])

        input_shape = updates.shape
        out_shape = [-1,
                     input_shape[1] * self.size[0],
                     input_shape[2] * self.size[1],
                     input_shape[3]]
        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )
        
class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def get_config(self):
        config = super().get_config().copy()
        config.update({"size":self.size})
        return config
        

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        mask = K.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')

        if output_shape is None:
            output_shape = (
                input_shape[0],
                input_shape[1] * self.size[0],
                input_shape[2] * self.size[1],
                input_shape[3])

        ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                            K.flatten(updates),
                            [K.prod(output_shape)])

        input_shape = updates.shape
        out_shape = [-1,
                     input_shape[1] * self.size[0],
                     input_shape[2] * self.size[1],
                     input_shape[3]]
        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3]
        )

class unet():
    def __init__(self, p=[(3,3), (1, 1), 'same', 'he_uniform', True, False]):
        self.params = dict(kernel_size=p[0], strides=p[1], padding=p[2], kernel_initializer=p[3])
        self.params_MUP = dict(kernel_size=p[0], padding=p[2])
        self.use_upsampling = p[-2]
        self.concat_axis = -1
        self.mup = p[-1]

    def __call__(self, input_imgs):
        self.inputs = input_imgs
        self.input_chn = np.shape(self.inputs)[-1] # 1 for gray scale, 3 for RGB
        self.input_row = np.shape(self.inputs)[1]
        self.input_col = np.shape(self.inputs)[2]
        self.input_shape = (self.input_row, self.input_col, self.input_chn)
        uNet = self.model(self.input_shape)
        return uNet

    def encConv2Pool(self, inputs, dim=64, g='relu', numLayer="first"):
        encode  = MaxPool2D(pool_size=(2, 2), name=f"en_pool_{numLayer}")(inputs)
        encode = Conv2D(filters=dim, activation=g, **self.params, name=f"en_conv_{numLayer}_1")(encode)
        encode = BatchNormalization()(encode)
        encode = Conv2D(filters=dim, activation=g, **self.params, name=f"en_conv_{numLayer}_2")(encode)
        encode = BatchNormalization()(encode)
        encode = Activation(g)(encode)
        return encode

    def encConv2MaxPool(self, inputs, dim=64, g='relu', numLayer="first"):
        encode = Conv2D(filters=dim, **self.params, name=f"en_conv_{numLayer}_1")(inputs)
        encode = BatchNormalization()(encode)
        encode = Activation(g)(encode)
        encode = Conv2D(filters=dim, **self.params, name=f"en_conv_{numLayer}_2")(encode)
        encode = BatchNormalization()(encode)
        encode = Activation(g)(encode)
        if numLayer=="L1" : encode = Dropout(0.2)(encode, training=True) # fine part
        elif numLayer=="L2" : encode = Dropout(0.2)(encode, training=True) # fine part
        elif numLayer=="L3" : encode = Dropout(0.3)(encode, training=True) # coarse part
        elif numLayer=="L4" : encode = Dropout(0.3)(encode, training=True) # coarse part
        elif numLayer=="L5" : encode = Dropout(0.6)(encode, training=True) # coarse part
        else : pass

        value, position = MaxPoolingWithArgmax2D()(encode) 
        return value, position

    def decCatConv2Up(self, inputs, concat_in, dim=64, numLayer="fourth"):
        if self.use_upsampling : 
            decode = UpSampling2D(name=f"de_up_{numLayer}",interpolation='bilinear')(inputs)
        else : 
            decode = Conv2DTranspose(filters=dim, **self.params_trans, name=f"de_convT_{numLayer}")(inputs)

        if concat_in : decode = concatenate([decode, concat_in], axis=self.concat_axis, name=f"de_cat_{numLayer}")
        print("concat_in : ", concat_in)
        decode = Conv2D(filters=dim, activation='relu', **self.params, name=f"de_conv_{numLayer}_1")(decode)
        encode = BatchNormalization()(decode)
        decode = Conv2D(filters=dim, activation='relu', **self.params, name=f"de_conv_{numLayer}_2")(decode)
        encode = BatchNormalization()(decode)
        return decode

    def decCatConv2UnPool(self, inputs, concat_in, g="relu", dim=64, numLayer="fourth"):
        unpool = MaxUnpooling2D()([inputs[0], inputs[1]])
        if numLayer == "L4": pass
        else : unpool = concatenate([unpool, concat_in],axis=-1)
        
        decode = Convolution2D(dim, **self.params_MUP, name=f"de_conv_{numLayer}_1")(unpool)
        decode = BatchNormalization()(decode)
        decode = Activation(g)(decode)
        
        decode = Convolution2D(dim, **self.params_MUP, name=f"de_conv_{numLayer}_2")(decode)
        decode = BatchNormalization()(decode)
        decode = Activation(g)(decode)
        
        decode = Convolution2D(dim, **self.params_MUP, name=f"de_conv_{numLayer}_3")(decode)
        decode = BatchNormalization()(decode)
        decode = Activation(g)(decode) 
        return decode    

    
    def model(self, imgs_shape=(400,400,1), msks_shape=1, dropout=0.2):
        num_chan_in  = imgs_shape[-1]
        num_chan_out = 1
        # num_chan_out = msks_shape[self.concat_axis] #-----> mask는 나중에 classification / segmentation 다룰 때 적용.

        inputs = Input(self.input_shape, name="OCTAmages")
        input_dim = 8

        if self.mup == False:
            ### encoder start
            # Layer 1    
            l1_encode = Conv2D(filters=input_dim, **self.params, name="en_conv_L1_1")(inputs)
            l2_encode = self.encConv2Pool(inputs=l1_encode, dim=input_dim*2,  g='sigmoid', numLayer="L2")   # Layer 2
            l3_encode = self.encConv2Pool(inputs=l2_encode, dim=input_dim*4,  g='sigmoid', numLayer="L3")   # Layer 3
            l4_encode = self.encConv2Pool(inputs=l3_encode, dim=input_dim*8,  g='sigmoid', numLayer="L4")   # Layer 4
            l5_encode = self.encConv2Pool(inputs=l4_encode, dim=input_dim*16, g='sigmoid', numLayer="L5")  # Layer 5            
            ### decoder start
            l4_decode = self.decCatConv2Up(inputs=l5_encode, concat_in=l4_encode, dim=input_dim*8,numLayer="L4") # Layer 4
            l3_decode = self.decCatConv2Up(inputs=l4_decode, concat_in=l3_encode, dim=input_dim*4,numLayer="L3") # Layer 3
            l2_decode = self.decCatConv2Up(inputs=l3_decode, concat_in=l2_encode, dim=input_dim*2,numLayer="L2") # Layer 2
            l1_decode = self.decCatConv2Up(inputs=l2_decode, concat_in=l1_encode, dim=input_dim*1,numLayer="L1") # Layer 1
        else: 
            v1 = Conv2D(filters=input_dim, **self.params, name="en_conv_L1_1")(inputs)
            (v2,p2) = self.encConv2MaxPool(inputs=v1, dim=input_dim*2,  g='sigmoid', numLayer="L2")   # Layer 2
            (v3,p3) = self.encConv2MaxPool(inputs=v2, dim=input_dim*4,  g='sigmoid', numLayer="L3")   # Layer 3
            (v4,p4) = self.encConv2MaxPool(inputs=v3, dim=input_dim*8,  g='sigmoid', numLayer="L4")   # Layer 4
            (v5,p5) = self.encConv2MaxPool(inputs=v4, dim=input_dim*16, g='sigmoid', numLayer="L5")  # Layer 5
            ### decoder start
            # v4_d = self.decCatConv2UnPool(inputs=(v5,p5),   concat_in=v4, dim=input_dim*8, numLayer="L4") # Layer 4
            v4_d = self.decCatConv2Up(inputs=v5, concat_in=False, dim=input_dim*8, numLayer="L4") # Layer 4
            v3_d = self.decCatConv2UnPool(inputs=(v4_d,p4), concat_in=v3, dim=input_dim*4, numLayer="L3") # Layer 3
            # v3_d = self.decCatConv2Up(inputs=v4, concat_in=False, dim=input_dim*4, numLayer="L3") # Layer 3
            v2_d = self.decCatConv2UnPool(inputs=(v3_d,p3), concat_in=v2, dim=input_dim*2, numLayer="L2") # Layer 2
            v1_d = self.decCatConv2UnPool(inputs=(v2_d,p2), concat_in=v1, dim=input_dim*1, numLayer="L1") # Layer 1
            l1_decode = v1_d

        # Output
        output = Conv2D(filters=num_chan_out, kernel_size=(1,1),activation="sigmoid", name="Output")(l1_decode)
        unet_model = Model(inputs=[inputs], outputs=[output], name="2DUNet")

        return unet_model

