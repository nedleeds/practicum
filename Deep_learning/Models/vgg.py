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
    def __init__(self):
        pass

    def __call__(self, input_imgs):
        self.inputs = input_imgs
        input_chn = np.shape(self.inputs)[-1] # 1 for gray scale, 3 for RGB
        input_row = np.shape(self.inputs)[1]
        input_col = np.shape(self.inputs)[2]
        input_shape = (input_row, input_col, input_chn)
         
        pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(input_row, input_col, 3))
        pre_trained_vgg.trainable = False
        pre_trained_vgg.summary()
        additional_model = models.Sequential()
        additional_model.add(pre_trained_vgg)
        additional_model.add(layers.Flatten())
        additional_model.add(layers.Dense(4096, activation='relu'))
        additional_model.add(layers.Dense(2048, activation='relu'))
        additional_model.add(layers.Dense(1024, activation='relu'))
        additional_model.add(layers.Dense(6, activation='softmax'))
        # additional_model.summary()
        return additional_model