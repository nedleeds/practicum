import os
import cv2
import numpy as np
from PIL import Image

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
            # im = Image.open(os.path.join(self.DATADIR, f))
            # img = im.crop()
            self.octa.append(img)
        return self.octa
