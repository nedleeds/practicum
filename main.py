import os
import cv2
import numpy as np 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from Data_processing    import dataload, display, normalizing
from Deep_learning      import train
from Filter             import filtering

n = 3

# octa = dataload(datadir="/root/Share/data/OCTA-500_gt/OCTA_6M/Projection Maps/OCTA(ILM_OPL)", outdir="./result")()
octa  = dataload(datadir="/root/Share/data/dataset/og",  outdir="/root/Share/data/result")()
mask  = dataload(datadir="/root/Share/data/dataset/bvm", outdir="/root/Share/data/result")()
label = dataload(datadir="/root/Share/data/dataset/label", outdir="/root/Share/data/result")()

# check    = display(mask)(numToShow=n, colormap='gray')
nmz_octa = normalizing(octa)(fromMinusOne=False, opt="max") # opt="max" : /255. , opt="minmax"
nmz_mask = normalizing(mask)(fromMinusOne=False, opt="max") # opt="max" : /255. , opt="minmax"
# check    = display((nmz_octa, label))(numToShow=n)
# check    = display((nmz_octa, label))(numToShow=n, opt='hist')

# # This is for frangi filtered(from MATLAB) => CLAHE + OTSU (PYTHON)
# img = cv2.imread(os.path.join("/root/Share/data/dataset/og","frangi_out.png"), cv2.IMREAD_GRAYSCALE)
# frangi_octa = filtering(img)('frangi')

# frangi_octa = filtering(nmz_octa)('frangi')
# check    = display(frangi_octa)(numToShow=n, colormap='gray')print(label)

img_label = {}
dr_dataset = {}
nc_dataset = {}

for k in nmz_octa.keys():
    if label[int(k)]=="DR" or label[int(k)]=="NORMAL":
        img_label[k]=[label[int(k)]]
        img_label[k].append(nmz_octa[k])

print(img_label)

# predicted = train('unet','segmentation')([nmz_octa, nmz_mask])
predicted = train('vgg','classification')(img_label)
# check    = display(predicted)(numToShow=n, colormap='gray')
