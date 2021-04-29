import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt 
plt.style.use('dark_background')


class display():
    def __init__(self, images):
        self.images = images
        self.len = len(self.images)
        try:
            self.row = np.shape(self.images)[1] 
            self.col = np.shape(self.images)[2]
        except:
            self.row = np.shape(self.images)[0]
            self.col = np.shape(self.images)[1] 
        self.idx = 0
    

    def __call__(self, idx=0, numToShow=1, colormap="gray", opt='default'): #idx should be int or list, opt:default-> 'nothing'
        self.idx = idx
        self.num = numToShow
        self.cmp = colormap.lower()
        self.opt = opt

        if self.len==1 : 
            if self.idx==False:
                plt.figure(figsize=(15,15))
                if self.opt.lower() =='hist' : plt.hist(self.images)
                else : plt.imshow(self.images, cmap='gray'), plt.axis(False)
            
            elif self.idx:
                plt.figure(figsize=(15,15))
                if self.opt.lower() =='hist' : plt.hist(self.images), plt.title(f'octa img[{self.idx}]')
                else : plt.imshow(self.images, cmap='gray'), plt.title(f'octa img[{self.idx}]'), plt.axis(False)
            
        elif self.len>1:
            if self.num>1:
                plt.figure(figsize=(15*self.num, 15))
                if idx==False:
                    #randomly select
                    idcs = np.random.randint(0, self.len, self.num)
                    for pos, i in enumerate(idcs):
                        if self.opt.lower()=='hist' :plt.subplot(1, self.num, pos+1), plt.hist(self.images[i]), plt.title(f'octa img[{i}]')
                        else : plt.subplot(1, self.num, pos+1), plt.imshow(self.images[i], cmap=self.cmp), plt.title(f'octa img[{i}]'), plt.axis(False)
                else:
                    for pos, i in enumerate(self.idx):
                        if self.opt.lower()=='hist' :plt.subplot(1, self.num, pos+1), plt.hist(self.images[i]), plt.title(f'octa img[{i}]')
                        else : plt.subplot(1, self.num, pos+1), plt.imshow(self.images[i], cmap=self.cmp), plt.title(f'octa img[{i}]'), plt.axis(False)
            else:
                # randomly selected one image
                plt.figure(figsize=(15, 15))
                i = np.random.randint(0, self.len, 1)
                if self.opt.lower()=='hist' : plt.hist(self.images[i]), plt.title(f'octa img[{i}]')
                else : plt.imshow(self.images[i], cmap=self.cmp), plt.title(f'octa img[{i}]'), plt.axis(False)
        else:
            print("you need to check the length of images")
            pass
        plt.show()
        plt.close('all')
        