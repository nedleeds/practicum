import numpy as np
import matplotlib.pyplot as plt
from .dataset import dataload

class normalizing():
    def __init__(self, images):
        self.images = images
        self.len = len(self.images)
        self.row = np.shape(self.images['10001'])[0]
        self.col = np.shape(self.images['10001'])[1]
        self.processed = {}

    def __call__(self, opt="minmax", fromMinusOne=False): # pos : 0~1, neg : -1~1
        keys=sorted(self.images.keys())
        for i in keys:
            if opt.lower() == "minmax":
                xmax, xmin = self.images[i].max(), self.images[i].min()
                self.images[i] = (self.images[i] - xmin)/(xmax - xmin) # min, max Normalizing : 0~1
                self.images[i] = self.images[i].astype(float)
                if fromMinusOne : self.images[i] = self.images[i]*2-1 # modify the range to [-1~1]
                else : pass
                self.processed[i] = self.images[i]
            elif opt.lower() == "max":
                self.images[i] = self.images[i]/255.
                self.images[i] = self.images[i].astype(float)
                self.processed[i] = self.images[i]
                if fromMinusOne : self.images[i] = self.images[i]*2-1 # modify the range to [-1~1]
                else : pass
                self.processed[i] = self.images[i]
            else:
                print("default option is 'minmax' you just put a wrong Normalizing optiong")
                return
        
        return self.processed
