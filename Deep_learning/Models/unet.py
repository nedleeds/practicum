import os
import numpy as np

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {'0', '1', '2'} ==> not for see the gpu contents
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose, concatenate, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

class unet():
    def __init__(self, p=[(3,3), (2,2), (1, 1), 'same', 'he_uniform', True]):
        self.params = dict(kernel_size=p[0], strides=p[2], padding=p[3], kernel_initializer=p[4])
        self.params_trans = dict(kernel_size=p[1], strides=p[2], padding=p[3])
        self.use_upsampling = p[5]
        self.concat_axis = -1

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

    def decCatConv2Up(self, inputs, concat_in, dim=64, numLayer="fourth"):
        if self.use_upsampling : 
            decode = UpSampling2D(name=f"de_up_{numLayer}",interpolation='bilinear')(inputs)
        else : 
            decode = Conv2DTranspose(filters=dim, **self.params_trans, name=f"de_convT_{numLayer}")(inputs)
    
        decode = concatenate([decode, concat_in], axis=self.concat_axis, name=f"de_cat_{numLayer}")
        decode = Conv2D(filters=dim, activation='relu', **self.params, name=f"de_conv_{numLayer}_1")(decode)
        encode = BatchNormalization()(decode)
        decode = Conv2D(filters=dim, activation='relu', **self.params, name=f"de_conv_{numLayer}_2")(decode)
        encode = BatchNormalization()(decode)
        return decode

    
    def model(self, imgs_shape=(400,400,1), msks_shape=1, dropout=0.2):
        num_chan_in  = imgs_shape[-1]
        num_chan_out = 1
        # num_chan_out = msks_shape[self.concat_axis] #-----> mask는 나중에 classification / segmentation 다룰 때 적용.

        inputs = Input(self.input_shape, name="OCTAmages")

        ### encoder start
        # Layer 1
        input_dim = 8
        l1_encode = Conv2D(filters=input_dim, **self.params, name="en_conv_L1_1")(inputs)
        l1_encode = Conv2D(filters=input_dim, **self.params, name="en_conv_L1_2")(inputs) 
        l2_encode = self.encConv2Pool(inputs=l1_encode, dim=input_dim*2,  g='sigmoid', numLayer="L2")   # Layer 2
        l3_encode = self.encConv2Pool(inputs=l2_encode, dim=input_dim*4,  g='sigmoid', numLayer="L3")   # Layer 3
        l4_encode = self.encConv2Pool(inputs=l3_encode, dim=input_dim*8,  g='sigmoid', numLayer="L4")   # Layer 4
        l5_encode = self.encConv2Pool(inputs=l4_encode, dim=input_dim*16, g='sigmoid', numLayer="L5")  # Layer 5
        
        ### decoder start
        l4_decode = self.decCatConv2Up(inputs=l5_encode, concat_in=l4_encode, dim=input_dim*8,numLayer="L4") # Layer 4
        l3_decode = self.decCatConv2Up(inputs=l4_decode, concat_in=l3_encode, dim=input_dim*4,numLayer="L3") # Layer 3
        l2_decode = self.decCatConv2Up(inputs=l3_decode, concat_in=l2_encode, dim=input_dim*2,numLayer="L2") # Layer 2
        l1_decode = self.decCatConv2Up(inputs=l2_decode, concat_in=l1_encode, dim=input_dim*1,numLayer="L1") # Layer 1

        # Output
        output = Conv2D(filters=num_chan_out, kernel_size=(1,1),activation="sigmoid", name="Output")(l1_decode)
        unet_model = Model(inputs=[inputs], outputs=[output], name="2DUNet")

        return unet_model