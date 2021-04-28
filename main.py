import cv2
import numpy as np 

from Data_processing        import dataload, display, normalizing
from Deep_learning          import train
from Image_processing       import curvelet, wavelet, ad, rg, fuzzy, eat, otsu, kMean

n = 3

octa     = dataload(datadir="./data/OCTA(ILM_OPL)", outdir="./result" )()
# check    = display(octa)(numToShow=n, colormap='gray') # you can set the indices for checking. eg) idx=[2,4,6] , Default : randomly select.

nmz_octa = normalizing(octa)(fromMinusOne=True, opt="minmax") # opt="max" : /255. , opt="minmax"
# check    = display(nmz_octa)(numToShow=n, opt='hist')

predicted = train('unet')(nmz_octa)
check    = display(predicted)(numToShow=3, colormap='gray')


# kMean_octa   = filter()('kMean')
# otsu_octa    = filter()('otsu')
# eat_octa     = filter()('eat')
# fuzzy_octa   = filter()('fuzzy')
# rg_octa      = filter()('rg')
# ad_octa      = filter()('ad')
# wvl_octa     = filter()('wvlt')
# cvl_octa     = filter()('cvlt')
# check = display(kMean_octa)(numToShow=3, opt='hist')