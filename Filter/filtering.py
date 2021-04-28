import os
import glob
import shutil
import SimpleITK as sitk
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from .frangi import frangi, hessian, utils

class filtering():
    def __init__(self, data):
        self.filter = ''
        self.data   = data
        self.niidir = "/root/Share/data/nii"

    def __call__(self, f):
        '''
        [  ]  1) K-mean's binarization
        [  ]  2) Otsu's
        [🔥️]  3) Frangi <= EAT : Enhaced Adaptive Frangi Filter
        [  ]  4) Fuzzy
        [  ]  5) Region Growing
        [  ]  6) Anisotropic Diffusion Filter
        [🔥️]  7) Wavelet
        [🔥️]  8) Curvelet
        '''
        
        self.filter = input("Choice Image Filter:\n(1) KMeans\n(2) Otsu\n(3) Frangi\n(4) Fuzzy\n\
                                                 \r(5) Region Growing\n(6) Anistropic Diffusion Filter\n\
                                                 \r(7) Wavelet\n(8) Curvelet : ")
        
        if   self.filter == '1': self.kmeans()
        elif self.filter == '2': self.otsu()
        elif self.filter == '3': self.frangi()
        elif self.filter == '4': self.fuzzy()
        elif self.filter == '5': self.rg()
        elif self.filter == '6': self.adf()
        elif self.filter == '7': self.wavelet()
        elif self.filter == '8': self.curvelet()
        else : return "You need to choose between (1)~(8)"

    def kmeans(self):
        print("{0:=^38}".format(" Kmeans "))

    def otsu(self):
        print("{0:=^38}".format(" Otsu "))

    def frangi(self):
        print("{0:=^38}".format(" Frangi "))
        octa_nii_list = sorted(glob.glob(os.path.join(self.niidir,'*')), key=os.path.getctime)
        octa_sitk     = sitk.ReadImage(octa_nii_list[0])
        octa_array    = sitk.GetArrayFromImage(octa_sitk)
        
        octa_frangi = frangi(octa_array[:,:,:,0], scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=True)
        
        plt.subplots(1,3,figsize=(10,30))
        plt.subplot(131), plt.imshow(octa_frangi[:,300,:], cmap='gray'), plt.axis(False)
        plt.subplot(132), plt.imshow(octa_frangi[:,350,:], cmap='gray'), plt.axis(False)
        plt.subplot(133), plt.imshow(octa_frangi[:,400,:], cmap='gray'), plt.axis(False)
        plt.show()

    def fuzzy(self):
        print("{0:=^38}".format(" Fuzzy "))

    def rg(self):
        print("{0:=^38}".format(" Region Growing "))

    def adf(self):
        print("{0:=^38}".format(" Anistropy Diffusion Filter "))

    def wavelet(self):
        print("{0:=^38}".format(" Wavelet "))

    def curvelet(self):
        print("{0:=^38}".format(" Curvelet "))

