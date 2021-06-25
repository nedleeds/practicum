import cv2
import random
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from .model_select import model_select, compile_train
from tensorflow.keras import metrics

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# from Models import unet, segnet
class train():
    def __init__(self,model_name='unet',kind='segmentation'):
        self.select = model_name
        self.kind = kind
        self.model_parameter = {
            'unet'   : [(3,3), (1, 1), 'same', 'he_uniform', True, True], # [ k, kT, s, p, i, upsample, MUP]
            'vgg'    : 2, # numbers of classes 
            'vae'    : [3, 2, 'same'] # [k, s, p]
        }
    def __call__(self, imgs):
        if self.kind =='classification':
            v = list(imgs.values())
            self.X, self.y = [], []
            for idx,(label, image) in enumerate(v) :
                self.X.append(image)
                self.y.append(label)
            self.dX, self.rX, self.cX = np.shape(self.X)
            self.ry, self.cy = np.shape(self.y)[0], 1
        else : 
            self.X = list(imgs[1].values())
            self.y = list(imgs[0].values())
            self.dX, self.rX, self.cX = np.shape(self.X)
            self.dy, self.ry, self.cy = np.shape(self.y)
        # self.X = x  # self.X -> image dictionaries - subject num : [octa images]
        # self.y = y  # self.y -> label dictionaries - subject num : [disease labels]

    
        train_valid = self.data_split()
        
        # model building
        # 1. select model
        # print("when call model_selece :", np.shape(self.train_X))
        
        selected_model = model_select(select=self.select)(self.train_X, self.model_parameter[self.select])
        selected_model.summary()
        # return
        # 2. compile model
        # print("when call compile_train :", np.shape(self.train_X))
        # compile_train(selected_model, self.select, train_valid)(opt='adam', epoch=500, batch=8, learn_r=0.001,
        #                                                         metric=[metrics.MeanSquaredError(),metrics.AUC()])
        compile_train(selected_model, self.select, train_valid)(opt='adam', epoch=50, batch=8, learn_r=0.01)
        # model prediction    
        if self.kind=='segmentation': model_out = self.savePredictedImg(selected_model)
        else : model_out = self.savePredictedClass(selected_model)
            

        # # 가중치 로드
        # model.load_weights(checkpoint_path)

        # # 모델 재평가
        # loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
        # print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

        return model_out

    def data_split(self):
        X_train,     self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, test_size=0.1, random_state=2)
        self.train_X, self.val_X, self.train_y, self.val_y  = train_test_split(X_train, self.train_y, test_size=0.1, random_state=4)

        self.train_X = tf.reshape(self.train_X, (-1, self.rX, self.cX, 1))
        self.val_X   = tf.reshape(self.val_X,   (-1, self.rX, self.cX, 1))
        self.test_X  = tf.reshape(self.test_X,  (-1, self.rX, self.cX, 1))

        if self.kind =='segmentation':    
            self.train_y = tf.reshape(self.train_y, (-1, self.rX, self.cX, 1))
            self.val_y   = tf.reshape(self.val_y,   (-1, self.rX, self.cX, 1))
            self.test_y  = tf.reshape(self.test_y,  (-1, self.rX, self.cX, 1))
        else:
            self.train_y = self.onehot_encoder(self.train_y)
            self.val_y   = self.onehot_encoder(self.val_y)
            self.test_y  = self.onehot_encoder(self.test_y)            

        train_valid = [(self.train_X, self.train_y), (self.val_X, self.val_y)]

        return train_valid

    def onehot_encoder(self, labels):        
        encoder = LabelEncoder()
        encoder.fit(labels)
        labels2 = encoder.transform(labels)
        oh_labels = tf.one_hot(labels2, len(set(labels)))
        # labels = labels.reshape(-1,1)
        # oh_encoder = OneHotEncoder()
        # oh_encoder.fit(labels)
        # oh_labels = oh_encoder.transform(labels)
        # oh_labels.toarray()
        return oh_labels

    def savePredictedImg(self,selected_m):
        for i in range(len(self.test_X)):
            model_out = []
            predicted = selected_m.predict((np.expand_dims(self.test_X[i],0)))
            model_out.append(np.reshape(predicted,(self.rX, self.cX)))
            plt.close('all')
            plt.subplots(1,3, figsize=(21,7))    
            plt.subplot(131), plt.imshow(np.reshape(self.test_X[i],(self.rX, self.cX)), cmap='gray'), plt.title("enface")
            plt.subplot(132), plt.imshow(predicted.reshape(self.rX, self.cX), cmap='gray'), plt.title("predicted")
            plt.subplot(133), plt.imshow(self.test_y[i], cmap='gray'), plt.title("ground truth")
            # # plt.show()
            plt.savefig('/root/Share/data/result/predict/predict_'+str(i)+'.png')
            return model_out
    
    def savePredictedClass(self,selected_m):
        predicted = selected_m.predict(self.test_X)
        correct = 0
        wrong = 0
        for idx, val in enumerate(predicted):
            if np.argmax(val)==1: pre="NORMAL"
            else: pre="DR"
            if np.argmax(self.test_y.numpy()[idx])==1: tst="NORMAL"
            else: tst="DR"
            print(f"pre/test:{pre}/{tst}")
            if pre==tst : correct+=1
            else : wrong +=1
        print(f"wrong / correct : {wrong} / {correct}")
        print(f"correct percentage : {round(correct/(wrong+correct),2)*100}%")
