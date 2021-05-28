import os
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


class dataload():
    def __init__(self, datadir, outdir):
        self.crops = []
        self.octa = []
        self.DATADIR = datadir
        self.OUTDIR = outdir
        self.width = 0
        self.height = 0

    def __call__(self):
        for f in os.listdir(self.DATADIR):
            img = cv2.imread(os.path.join(self.DATADIR, f), cv2.IMREAD_GRAYSCALE)
            center_x, center_y = int(np.shape(img)[0]/2), int(np.shape(img)[1]/2)
            croped = img[center_x-128:center_x+128, center_y-128:center_y+128]
            self.octa.append(img)
        return self.octa
