import cv2
import random
import numpy as np
import tensorflow as tf

from .metrics import metric
from sklearn.model_selection import train_test_split
from .model_select           import model_select, compile_train
from tensorflow.keras import metrics

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# from Models import unet, segnet
class train():
    def __init__(self,model_name='unet'):
        self.select = model_name
        self.model_parameter = {
            'unet' : [(3,3), (2,2), (1, 1), 'same', 'he_uniform', True], # [ k, kT, s, p, i, upsample ]
            'segnet' : [(3,3), (1, 1), 'same', 'he_uniform', True], # [ k, s, p, i, upsample ]
            'vae' : [3, 2, 'same'] # [k, s, p]
        }
    def __call__(self, imgs):
        self.X = imgs[0]
        self.y = imgs[1]
        self.dX, self.rX, self.cX = np.shape(self.X)
        self.dy, self.ry, self.cy = np.shape(self.y)
        
        train_valid = self.data_split()
        
        # model building
        # 1. select model
        # print("when call model_selece :", np.shape(self.train_X))
        selected_model = model_select(select=self.select)(self.train_X, self.model_parameter[self.select])
        selected_model.summary()

        # 2. compile model
        # print("when call compile_train :", np.shape(self.train_X))
        compile_train(selected_model, self.select, train_valid)(opt='adam', epoch=100, 
                                                                batch=32, learn_r=0.001,
                                                                metric=[metrics.MeanSquaredError(),metrics.AUC()])
        
        # model prediction
        model_out = []
        
        
        for i in range(len(self.test_X)):
            predicted = selected_model.predict((np.expand_dims(self.test_X[i],0)))
            model_out.append(np.reshape(predicted,(self.rX, self.cX)))
            plt.close('all')
            plt.subplots(1,3, figsize=(21,7))    
            plt.subplot(131), plt.imshow(np.reshape(self.test_X[i],(self.rX, self.cX)), cmap='gray'), plt.title("enface")
            plt.subplot(132), plt.imshow(predicted.reshape(self.rX, self.cX), cmap='gray'), plt.title("predicted")
            plt.subplot(133), plt.imshow(self.test_y[i], cmap='gray'), plt.title("ground truth")
            # # plt.show()
            plt.savefig('/root/Share/data/result/predict/predict_'+str(i)+'.png')
            
        

        # # 가중치 로드
        # model.load_weights(checkpoint_path)

        # # 모델 재평가
        # loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
        # print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

        return model_out

    def data_split(self):
        X_train, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X_train, self.train_y, test_size=0.3, random_state=42)

        self.train_X = tf.reshape(self.train_X, (-1, self.rX, self.cy, 1))
        self.train_y = tf.reshape(self.train_y, (-1, self.rX, self.cy, 1))
        self.val_X   = tf.reshape(self.val_X,   (-1, self.rX, self.cy, 1))
        self.val_y   = tf.reshape(self.val_y,   (-1, self.rX, self.cy, 1))
        self.test_X  = tf.reshape(self.test_X,  (-1, self.rX, self.cy, 1))
        self.test_y  = tf.reshape(self.test_y,  (-1, self.rX, self.cy, 1))

        train_valid = [(self.train_X, self.train_y), (self.val_X, self.val_y)]
        return train_valid
        

