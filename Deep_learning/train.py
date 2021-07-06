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
    def __init__(self, model_name='unet',kind='segmentation', class_num=2):
        self.select = model_name
        self.kind = kind
        self.model_parameter = {
            'unet'   : [(3,3), (1, 1), 'same', 'he_uniform', True, True], # [ k, kT, s, p, i, upsample, MUP]
            'vgg'    : class_num, # numbers of classes 
            'vae'    : [3, 2, 'same'] # [k, s, p]
        }
        self.X, self.y = [],[]

    def __call__(self, imgs): # imgs -> image, label split. 
        # model building

        seeds = [3]
        for seed in seeds :  
            # 1. select model    
            # s = 1
            train_valid = self.data_split(imgs, seed)
            selected_model = model_select(select=self.select)(self.train_X, self.model_parameter[self.select], 1)
            selected_model.summary()

            # 2. compile model
            compile_train(selected_model, self.select, train_valid)(opt='adam', epoch=100, batch=8, learn_r=0.001)
            
            # 3. model prediction    
            if self.kind=='segmentation': model_out = self.saveTestImg(selected_model)
            else : model_out = self.testScore(selected_model)

            # this is for "Binary class"
            train_dr = np.sum(self.train_y,0)[0]
            train_nc = np.sum(self.train_y,0)[1]
            test_dr = np.sum(self.test_y,0)[0]
            test_nc = np.sum(self.test_y,0)[1]

            '''
            with open('seed_acc.txt', 'a') as f:
                f.write(f"seed-{seed}\n")
                f.write(f"\tacc:{model_out[0]}, precision:{model_out[1]}, recall:{model_out[2]}, ")
                f.write(f"TP/TN/FP/FN:{model_out[-1][0]}/{model_out[-1][1]}/{model_out[-1][2]}/{model_out[-1][3]} \n")
                f.write(f"\tTrain\tset - Normal/DR : {int(train_nc)}/{int(train_dr)}\n")
                f.write(f"\tTest\tset - Normal/DR : {int(test_nc)}/{int(test_dr)}\n\n")
            '''
            if model_out[0] > 87.0:
                print(f"seed : {seed}, acc:{model_out}")

        return model_out

    def data_split(self, imgs, seed):
        if self.kind =='classification':
            # class_num = [len(args[x]) for x in range(len(args))]
            # for i in range(len(args)):
            #     for label, image in list(*args.values()):
            #         self.X.append(image)
            #         self.y.append(label)
            for label, image in list(imgs.values()) :
                self.X.append(image)
                self.y.append(label)
            self.dX, self.rX, self.cX = np.shape(self.X)
            self.ry, self.cy = np.shape(self.y)[0], 1
        else : 
            self.X = list(imgs[1].values())
            self.y = list(imgs[0].values())
            self.dX, self.rX, self.cX = np.shape(self.X)
            self.dy, self.ry, self.cy = np.shape(self.y)

        self.train_X, self.test_X, self.train_y, self.test_y  = train_test_split(self.X, self.y, test_size=0.2, random_state=seed)
        # X_train,     self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, test_size=0.1, random_state=2)
        # self.train_X, self.val_X, self.train_y, self.val_y  = train_test_split(X_train, self.train_y, test_size=0.15, random_state=4)

        self.train_X = tf.reshape(self.train_X, (-1, self.rX, self.cX, 1))
        # self.val_X   = tf.reshape(self.val_X,   (-1, self.rX, self.cX, 1))
        self.test_X  = tf.reshape(self.test_X,  (-1, self.rX, self.cX, 1))

        if self.kind =='segmentation':    # image reshaping
            self.train_y = tf.reshape(self.train_y, (-1, self.rX, self.cX, 1))
            # self.val_y   = tf.reshape(self.val_y,   (-1, self.rX, self.cX, 1))
            self.test_y  = tf.reshape(self.test_y,  (-1, self.rX, self.cX, 1))
        else: # labeling --> need to onehot encoding
            self.train_y = self.onehot_encoder(self.train_y)
            # self.val_y   = self.onehot_encoder(self.val_y)
            self.test_y  = self.onehot_encoder(self.test_y)            

        # train_valid = [(self.train_X, self.train_y), (self.val_X, self.val_y)]
        train_valid = [self.train_X, self.train_y]
        return train_valid

    def onehot_encoder(self, labels):        
        encoder = LabelEncoder()
        encoder.fit(labels)
        labels2 = encoder.transform(labels)
        oh_labels = tf.one_hot(labels2, len(set(labels)))
        return oh_labels

    def saveTestImg(self,selected_m):
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

    def testScore(self,selected_m):
        predicted = selected_m.predict(self.test_X)
        correct   = 0
        wrong     = 0
        confuse   = []

        for idx, val in enumerate(predicted):
            if np.argmax(val)==1: pre="NORMAL"
            else: pre="DR"
            if np.argmax(self.test_y.numpy()[idx])==1: gt="NORMAL"
            else: gt="DR"
            print(f"pre/test:{pre}/{gt}")
            if pre==gt : correct+=1
            else : wrong +=1
            
            if   gt=="NORMAL" and pre=="NORMAL" : confuse.append("TP")
            elif gt=="NORMAL" and pre=="DR"     : confuse.append("FN")
            elif gt=="DR"     and pre=="NORMAL" : confuse.append("FP")
            elif gt=="DR"     and pre=="DR"     : confuse.append("TN")
            else: return "wrong operation chech testScore again"
        
        tp = confuse.count("TP")
        fn = confuse.count("FN")
        fp = confuse.count("FP")
        tn = confuse.count("TN")

        acc   = round((tp+tn)/(tp+tn+fp+fn),3)
        prcsn = round(tp/(tp+fp),3)
        rcll  = round(tp/(tp+fn),3)
        print(f"test : acc-{acc}, precision-{prcsn}, recall={rcll}, TP/TN/FP/FN-{tp}/{tn}/{fp}/{fn}")
        # print(f"wrong / correct : {wrong} / {correct}")
        # print(f"correct percentage : {round(correct/(wrong+correct),2)*100}%")
        # acc = round(correct/(wrong+correct),3)*100
        return [acc*100, prcsn, rcll, [tp,tn,fp,fn]]