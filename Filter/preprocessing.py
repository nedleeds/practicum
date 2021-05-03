import numpy as np
import matplotlib.pyplot as plt

class normalizing():
    def __init__(self, images):
        self.images = images
        self.len = len(images)
        self.dim = len(images.shape)
        
        if self.dim==3:
            self.row = np.shape(self.images)[1]
            self.col = np.shape(self.images)[2]
        elif self.dim==2 :
            self.row = np.shape(self.images)[0]
            self.col = np.shape(self.images)[1]
        else:
            return "dimension is digger than 3. check again."
        self.processed = []

    def __call__(self, opt="minmax", fromMinusOne=False): # pos : 0~1, neg : -1~1
        if opt.lower() == "minmax":
            for img in self.images:
                xmax, xmin = img.max(), img.min()
                img = (img - xmin)/(xmax - xmin) # min, max Normalizing : 0~1
                img = img.astype(float)
                if fromMinusOne : img = img*2-1 # modify the range to [-1~1]
                else : pass
                self.processed.append(img)
            else: pass

        elif opt.lower() == "max":
            for img in self.images:
                img = img/255.
                img = img.astype(float)
                self.processed.append(img)
                if fromMinusOne : img = img*2-1 # modify the range to [-1~1]
                else : pass
                self.processed.append(img)

        else:
            print("default option is 'minmax' you just put a wrong Normalizing optiong")
            pass 
        
        return self.processed

# normalizing(opt="minmax", fromMinusOne=False)