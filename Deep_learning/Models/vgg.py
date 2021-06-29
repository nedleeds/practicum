from keras import models, layers
from keras.applications import VGG16
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
 
 
class vgg():
    def __init__(self,classnum=6):
        self.classnum=classnum

    def __call__(self, input_imgs):
        self.inputs = input_imgs
        input_chn = np.shape(self.inputs)[-1] # 1 for gray scale, 3 for RGB
        input_row = np.shape(self.inputs)[1]
        input_col = np.shape(self.inputs)[2]

        input_shape = (input_row, input_col, input_chn)
        model = models.Sequential()
        
        model.add(Input(shape=input_shape, name="gray_input"))
        model.add(layers.Conv2D(3, (1, 1), padding='same', activation='relu', name="rgb_input"))
        pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(input_row, input_col, 3))
        pre_trained_vgg.trainable = True
        model.add(pre_trained_vgg)
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        # model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2048, activation='relu'))
        # model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
        # model.add(layers.Dropout(0.1))
        model.add(layers.Dense(self.classnum, activation='softmax'))
        
        return model