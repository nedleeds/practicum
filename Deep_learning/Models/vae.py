import imageio
import glob
import time
import cv2
import tensorflow as tf
import os
from tensorflow.keras import layers
from IPython import display

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
from tensorflow import keras

class vae():
    def __init__(self, p=[3, 2, 'same']):
        self.params = dict(kernel_size=p[0], strides=p[1], padding=p[2])

    def __call__(self, input_imgs):
        self.inputs = input_imgs        
        self.input_dim = np.shape(self.inputs)[0]
        self.input_row = np.shape(self.inputs)[1]
        self.input_col = np.shape(self.inputs)[2]
        self.input_chn = np.shape(self.inputs)[-1] # 1:gray, 3:RGB
        self.input_shape = (self.input_row, self.input_col, self.input_chn)
        uNet = self.model(self.input_shape)
        return uNet
        
    def encoder(self, input_encoder):
        inputs = keras.Input(shape=input_encoder, name='input_layer')
        # Block-1
        x = layers.Conv2D(32, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.LeakyReLU(name='lrelu_1')(x)
        # Block-2
        x = layers.Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.LeakyReLU(name='lrelu_2')(x)
        # Block-3
        x = layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.LeakyReLU(name='lrelu_3')(x)
        # Block-4
        x = layers.Conv2D(64, 3, 1, padding='same', name='conv_4')(x)
        x = layers.BatchNormalization(name='bn_4')(x)
        x = layers.LeakyReLU(name='lrelu_4')(x)
        # Final Block
        flatten = layers.Flatten()(x)
        mean = layers.Dense(2, name='mean')(flatten)
        log_var = layers.Dense(2, name='log_var')(flatten)
        model = tf.keras.Model(inputs, (mean, log_var), name="Encoder")
        return model

    def sampling(self, input_1,input_2):
        mean = keras.Input(shape=input_1, name='input_layer1')
        log_var = keras.Input(shape=input_2, name='input_layer2')
        out = layers.Lambda(sampling_reparameterization_model, name='encoder_output')([mean, log_var])
        enc_2 = tf.keras.Model([mean,log_var], out,  name="Encoder_2")
        return enc_2

    def decoder(self, input_decoder):
        inputs = keras.Input(shape=input_decoder, name='input_layer')
        x = layers.Dense(3136, name='dense_1')(inputs)
        x = layers.Reshape((7, 7, 64), name='Reshape_Layer')(x)
        # Block-1
        x = layers.Conv2DTranspose(64, 3, strides= 1, padding='same',name='conv_transpose_1')(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.LeakyReLU(name='lrelu_1')(x)
        # Block-2
        x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.LeakyReLU(name='lrelu_2')(x)
        # Block-3
        x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_3')(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.LeakyReLU(name='lrelu_3')(x)
        # Block-4
        outputs = layers.Conv2DTranspose(1, 3, 1,padding='same', activation='sigmoid', name='conv_transpose_4')(x)
        model = tf.keras.Model(inputs, outputs, name="Decoder")
        return model

    optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
    def mse_loss(self, y_true, y_pred):
        r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
        return 1000 * r_loss

    def kl_loss(sefl, mean, log_var):
        kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
        return kl_loss

    def vae_loss(self, y_true, y_pred, mean, var):
        r_loss = mse_loss(y_true, y_pred)
        kl_loss = kl_loss(mean, log_var)
        return  r_loss + kl_loss


    @tf.function
    def train_step(images):
        with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
            mean, log_var = enc(images, training=True)
            latent = sampling([mean, log_var])
            generated_images = dec(latent, training=True)
            loss = vae_loss(images, generated_images, mean, log_var)

        gradients_of_enc = encoder.gradient(loss, enc.trainable_variables)
        gradients_of_dec = decoder.gradient(loss, dec.trainable_variables)
        optimizer.apply_gradients(zip(gradients_of_enc, enc.trainable_variables))
        optimizer.apply_gradients(zip(gradients_of_dec, dec.trainable_variables))
        return loss

    def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
        train_step(image_batch)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    train(train_dataset, epoch)

    figsize = 15
    m, v = enc.predict(x_test[:25])
    latent = sampling([m,v])
    reconst = dec.predict(latent)
    fig = plt.figure(figsize=(figsize, 10))
    for i in range(25):
        ax = fig.add_subplot(5, 5, i+1)
        ax.axis('off')
        ax.text(0.5, -0.15, str(label_dict[y_test[i]]), fontsize=10, ha='center', transform=ax.transAxes)
        ax.imshow(reconst[i, :,:,0]*255, cmap = 'gray')
