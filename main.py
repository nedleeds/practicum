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

for k in nmz_octa.keys():
    if label[int(k)]=="DR" or label[int(k)]=="NORMAL":
        img_label[k]=[label[int(k)]]
        img_label[k].append(nmz_octa[k])

basepath = "/root/Share/data/dataset/disease"
ogpath = "/root/Share/data/dataset/og"
drpath = os.path.join(basepath, "DR")
ncpath = os.path.join(basepath, "NORMAL")
ampath = os.path.join(basepath, "AMD")
cnpath = os.path.join(basepath, "CNV")
cspath = os.path.join(basepath, "CSC")
rvpath = os.path.join(basepath, "RVO")
otpath = os.path.join(basepath, "OTHERS")

dr_dataset = {}
nc_dataset = {}
am_dataset = {}
cn_dataset = {}
cs_dataset = {}
rv_dataset = {}
ot_dataset = {}

for i in nmz_octa.keys():
    k = label[int(i)]
    v = cv2.imread(os.path.join(ogpath, str(i)+".bmp"), cv2.IMREAD_GRAYSCALE)
    if k=="DR":
        cv2.imwrite(os.path.join(drpath, i+"_DR.png"), v)
    elif k=="NORMAL":
        nc_dataset[i]=nmz_octa[i]
        cv2.imwrite(os.path.join(ncpath, i+"_NORMAL.png"), v)
    elif k=="AMD":
        am_dataset[i]=nmz_octa[i]
        cv2.imwrite(os.path.join(ampath, i+"_AMD.png"), v)
    elif k=="CSC":
        cs_dataset[i]=nmz_octa[i]
        cv2.imwrite(os.path.join(cspath, i+"_CSC.png"), v)
    elif k=="CNV":
        cn_dataset[i]=nmz_octa[i]
        cv2.imwrite(os.path.join(cnpath, i+"_CNV.png"), v)
    elif k=="RVO":
        rv_dataset[i]=nmz_octa[i]
        cv2.imwrite(os.path.join(rvpath, i+"_RVO.png"), v)
    elif k=="OTHERS":
        ot_dataset[i]=nmz_octa[i]
        cv2.imwrite(os.path.join(otpath, i+"_OTHERS.png"), v)
    else: pass

print(img_label)

# predicted = train('unet','segmentation')([nmz_octa, nmz_mask])
predicted = train('vgg','classification')(img_label)
# check    = display(predicted)(numToShow=n, colormap='gray')
