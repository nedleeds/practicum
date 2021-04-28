import cv2
import random
import numpy as np
import tensorflow as tf

from .metrics import metric
from sklearn.model_selection import train_test_split
from .model_select           import model_select, compile_train



# from Models import unet, segnet
class train():
    def __init__(self,model_name='unet'):
        self.select = model_name
        self.model_parameter = {
            'unet' : [(3,3), (2,2), (1, 1), 'same', 'he_uniform', True] # [ k, kT, s, p, i, upsample ]
            # [ kernel size, ConvT's kernel size, stride size , padding option, weight initialization, upsample(T/F)]
        }
    def __call__(self, imgs):
        self.d, self.r, self.c = np.shape(imgs)
        self.train, self.test  = train_test_split(imgs, test_size=0.1, shuffle=True)
        self.train, self.valid = train_test_split(self.train, test_size=0.3, shuffle=True)
        
        print(np.shape(self.train), np.shape(self.valid), np.shape(self.test))

        train = tf.reshape(self.train,(-1, self.r, self.c, 1))
        valid = tf.reshape(self.valid,(-1, self.r, self.c, 1))
        test  = tf.reshape(self.test ,(-1, self.r, self.c, 1))
        
        train_valid = [train, valid]
        
        # model building
        # 1. select model
        selected_model = model_select(select=self.select)(train, self.model_parameter[self.select])
        selected_model.summary()

        # 2. compile model
        compile_train(selected_model, self.select, train_valid)(opt='Adam', lss='mse', epoch=50, batch=4, learn_r=0.01)
        
        # model prediction
        model_out = []
        for i in range(len(test)):
            a = selected_model(tf.reshape(self.test[i],(1, self.r, self.c, 1)))
            model_out.append(np.reshape(a,(self.r,self.c)))

        cv2.imwrite('./result/predict.png', model_out[0])

        # # 가중치 로드
        # model.load_weights(checkpoint_path)

        # # 모델 재평가
        # loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
        # print("복원된 모델의 정확도: {:5.2f}%".format(100*acc))

        return model_out


