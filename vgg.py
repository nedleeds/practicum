from keras import models, layers
from keras.applications import VGG16
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
from keras.layers import Concatenate
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
 
 
class vgg():
    def __init__(self):
        gray_input = Input(shape=(400, 400, 1), dtype='float32', name='gray_input')
        gray_concat = Concatenate()([gray_input, gray_input, gray_input])

        pre_trained_vgg = VGG16(include_top=False, weights='imagenet', input_tensor=gray_concat)

        pre_trained_vgg.trainable = False # fix the weight of pre_trained vgg.
        
        additional_model = models.Sequential()  # making sequential model
        additional_model.add(pre_trained_vgg)   # adding pre-trained vgg
        additional_model.add(layers.Flatten())  # adding flatten layer
        additional_model.add(layers.Dense(4096, activation='relu'))
        additional_model.add(layers.Dense(2048, activation='relu'))
        additional_model.add(layers.Dense(1024, activation='relu'))
        additional_model.add(layers.Dense(3, activation='softmax')) # nc, dr, nodr 
        # additional_model.summary()
        
        
        # checkpoint = ModelCheckpoint(filepath='pretrained_VGG_weight.hdf5', 
        #             monitor='loss', 
        #             mode='min', 
        #             save_best_only=True)
        
        additional_model.compile(loss='categorical_crossentropy',
                                optimizer=optimizers.RMSprop(lr=2e-5),
                                metrics=['acc'])
        
        
        # history = additional_model.fit_generator(train_generator, 
        #             steps_per_epoch=math.ceil(train_generator.n / train_generator.batch_size), 
        #             epochs=300, 
        #             validation_data=val_generator, 
        #             validation_steps=math.ceil(val_generator.n / val_generator.batch_size), 
        #             callbacks=[checkpoint])
        
        # # number of train & validation samples 
        # # print(train_generator.n)
        # # print(val_generator.n)
        
        # # number of train & val batch_size
        # # print(train_generator.batch_size)
        # # print(val_generator.batch_size)
        
        
        # acc = history.history['acc']
        # val_acc = history.history['val_acc']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        
        # epochs = range(1, len(acc) + 1)
        
        # plt.plot(epochs, acc, 'b', label='Training acc')
        # plt.plot(epochs, val_acc, 'r', label='Validation acc')
        # plt.title('Accuracy')
        # plt.legend()
        # plt.figure()
        
        # plt.plot(epochs, loss, 'b', label='Training loss')
        # plt.plot(epochs, val_loss, 'r', label='Validation loss')
        # plt.title('Loss')
        # plt.legend()
        
        # plt.show()
