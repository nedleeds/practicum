import cv2
import numpy as np 
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from Data_processing    import dataload, display, normalizing
from Deep_learning      import train
from Filter             import filtering

n = 3

octa = dataload(datadir="./data/dataset/og", outdir="./result")()
mask = dataload(datadir="./data/dataset/bvm", outdir="./result")()
# check    = display(mask)(numToShow=n, colormap='gray')

nmz_octa = normalizing(octa)(fromMinusOne=False, opt="max") # opt="max" : /255. , opt="minmax"
nmz_mask = normalizing(mask)(fromMinusOne=False, opt="max") # opt="max" : /255. , opt="minmax"
# check    = display(nmz_octa)(numToShow=n, opt='hist')

# frangi_octa = filtering(nmz_octa)('frangi')
# check    = display(frangi_octa)(numToShow=n, colormap='gray')

predicted = train('unet')((nmz_octa, nmz_mask))
# check    = display(predicted)(numToShow=n, colormap='gray')
