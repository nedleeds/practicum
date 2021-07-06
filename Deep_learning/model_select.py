import os
import shutil
import tensorflow as tf 
import numpy as np

from .Models import unet, vgg
from keras   import optimizers

import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

class model_select(object):
    def __init__(self, select='unet'):
        self.select = select

    def __call__(self, input_images, params, drop_seed):
        if self.select.lower()=='unet':
            model = unet(params)(input_images)
        elif self.select.lower()=='vgg':
            model = vgg(params)(input_images, drop_seed)
        # elif self.select.lower()=='vae':
        #     model = vae(params)(input_images)
        else : pass

        return model

class compile_train():
    def __init__(self, selected_model, name, data):
        self.model_ = selected_model
        self.name   = name
        
        self.train_X  = data[0]
        self.train_y  = data[1]
        # self.train_X  = data[0][0]
        # self.train_y  = data[0][1]
        # self.val_X  = data[1][0]
        # self.val_y  = data[1][1]

    def __call__(self, opt='Adam', lss='mse', epoch=100, batch=8, learn_r=0.001):
        self.optimizer   = opt  
        self.loss        = self.myloss(self.train_X, self.train_y)

        if self.metrics() : 
            self.model_.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics())
        else:
            adam = optimizers.Adam(lr=learn_r)
            self.model_.compile(loss=self.loss, optimizer=adam)
        
        self.model_.fit(self.train_X, self.train_y,
                        batch_size=batch,
                        epochs=epoch, 
                        verbose=2
                        # callbacks=self.get_callbacks()
                        # validation_data=(self.val_X, self.val_y),
                        ) 
        
        return self.model_

    def get_callbacks(self):
        CURRDIR = os.getcwd()
        try:
            CHCKDIR = os.path.join(CURRDIR, f"Deep_learning/checkpoints/{self.name}")
            LOGSDIR = os.path.join(CURRDIR, f"Deep_learning/logs/{self.name}")
        except:
            os.mkdir(CHCKDIR)
            os.mkdir(LOGSDIR)

        if os.path.isdir(CHCKDIR):
            shutil.rmtree(CHCKDIR)
        if os.path.isdir(LOGSDIR):
            shutil.rmtree(LOGSDIR)
        if not os.path.isdir(CHCKDIR): 
            os.mkdir(CHCKDIR)
        if not os.path.isdir(LOGSDIR):
            os.mkdir(LOGSDIR)
        else:
            pass

        print("LOGSDIR = {}".format(LOGSDIR))
        print("CHCKDIR = {}".format(CHCKDIR))

        # early_stopping  = K.callbacks.EarlyStopping(patience=50, restore_best_weights=True), 
        model_ckpt      = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CHCKDIR,'model.{epoch:02d}-{loss:.2f}.h5'),
                                                             save_best_only=True ),
        board_ckpt      = tf.keras.callbacks.TensorBoard(log_dir = LOGSDIR)
        # model_callbacks = [early_stopping, model_ckpt, board_ckpt]
        model_callbacks = [model_ckpt, board_ckpt]
        
        return model_callbacks

    def myloss(self, y, y_hat):
        # loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # loss2 = tf.keras.losses.CategoricalHinge()
        # loss3 = tf.keras.losses.CosineSimilarity() # nope.
        loss4 = tf.keras.losses.CategoricalCrossentropy() # for disease classification
        total_loss = loss4
        return total_loss

    def metrics(self):
        m = [TruePositives(), TrueNegatives(), Precision(), Recall()]
        return m
