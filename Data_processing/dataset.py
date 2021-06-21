import os
import cv2
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import pandas as pd

class dataload():
    def __init__(self, datadir, outdir):
        self.crops = []
        self.octa = {}
        self.DATADIR = datadir
        self.OUTDIR = outdir
        self.width = 0
        self.height = 0

    def __call__(self):
        if self.DATADIR.split('/')[-1]!='label':
            flist = sorted(glob.glob(os.path.join(self.DATADIR, '*')), key=os.path.getctime)
            for f in flist:
                img = cv2.imread(os.path.join(f), cv2.IMREAD_GRAYSCALE)
                center_x, center_y = int(np.shape(img)[0]/2), int(np.shape(img)[1]/2)
                croped = img[center_x-128:center_x+128, center_y-128:center_y+128]
                num = f.split('.')[0].split('/')[-1]
                self.octa[num] = img
            return self.octa
        else:
            # extracting label from .csv file from ground truth data.
            xlsx_path = os.path.join(self.DATADIR, "6mm/labels.xlsx")
            data = pd.read_excel(xlsx_path, engine='openpyxl')[:300]
            print(data)
            id_num = data['ID'][:300]
            disease = data['Disease'][:300]
            label = {}
            for i, d in zip(id_num.astype(int), disease):
                label[i] = d
            return label
        
