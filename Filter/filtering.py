import os
import glob
import shutil
import SimpleITK as sitk
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import nibabel as nib
import numpy as np
import cv2
from scipy import ndimage, misc

from .frangi                 import frangi as frangi3d
from .frangi                 import hessian, utils
from skimage.filters         import frangi as frangi2d
from medpy.filter.smoothing  import anisotropic_diffusion
from .preprocessing           import normalizing
from PIL import Image
class filtering():
    def __init__(self, data=''):
        self.filter = ''
        self.datas   = data
        self.niidir = "/root/Share/data/nii"

    def __call__(self, f):
        '''
        [  ]  1) K-mean's binarization
        [OK]  2) Otsu's
        [OK]  3) Frangi
        [  ]  4) Fuzzy
        [  ]  5) Region Growing
        [OK]  6) Anisotropic Diffusion Filter
        [🔥️]  7) Wavelet
        [🔥️]  8) Curvelet
        '''
        # These code for making data.
        datadir   = "./data/OCTA(ILM_OPL)"
        ogdir     = "/root/Share/data/dataset/og"
        clahedir  = "/root/Share/data/dataset/clahe"
        bvmdir    = "/root/Share/data/dataset/bvm"
        resultdir = "/root/Share/data/result"

        niipath   = "/root/Share/data/nii/10001_OCTA_seg.nii.gz"
        dns_nii_path = os.path.join(resultdir, "10001_dns_octa.nii.gz")

        # nii = nib.load(niipath)
        # print(nii.get_fdata())
        # nii_arr = np.asarray(nii.get_fdata())
        # print(nii_arr.shape)

        dns_octa = nib.load(dns_nii_path)
        frg_octa = frangi3d(dns_octa, scale_range=(1, 10), scale_step=0.001, alpha=0.5, beta=0.5, frangi_c=500, black_vessels=False)
        nib.save(frg_octa, os.path.join(resultdir, "frangi3d.nii.gz"))

        ## processing all the images in directory 
        # for f in sorted(glob.glob(datadir+"/*"), key=os.path.getctime):
        #     octa = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
        #     octa_nm = normalizing(octa)(opt="max", fromMinusOne=False)
        #     octa_adf    = self.adf(np.array(octa_nm), _niter=2, _kappa=90, _gamma=0.1, _voxelspacing=None, _option=3)
        #     octa_frangi = self.frangi(octa_adf, _sigmas=(0,1), _scale_step=0.1, _black_ridges=False)
        #     octa_clahe  = self.CLAHE(octa_frangi*(1e+4))
        #     octa_bmask  = self.otsu(octa_clahe)
        #     cv2.imwrite(os.path.join(ogdir,f.split('/')[-1]), octa)
        #     cv2.imwrite(os.path.join(clahedir,f.split('/')[-1]), octa_clahe)
        #     cv2.imwrite(os.path.join(bvmdir, f.split('/')[-1]), octa_bmask*255)
        #     self.display(octa_nm, octa_adf, octa_frangi, octa_clahe, octa_bmask)

        ## adapting filtering to the image
        # for i in range(len(self.datas)):
        #     octa_nm     = (self.datas[i]*255).astype(np.uint8)
        #     octa_adf    = self.adf(octa_nm, _niter=2, _kappa=90, _gamma=0.1, _voxelspacing=None, _option=3)
        #     octa_frangi = self.frangi(octa_adf, _sigmas=1, _scale_step=0.001, _black_ridges=False)
        #     octa_clahe  = self.CLAHE(octa_frangi)
        #     # octa_bmi    = self.binarymedian(octa_clahe)
        #     octa_bmask  = self.otsu(octa_clahe)
            
        #     # plt.subplots(1,2, figsize=(20,10))
        #     # plt.subplot(121), plt.imshow(octa_clahe, cmap='gray'), plt.title('CLAHE')
        #     # plt.subplot(122), plt.imshow(octa_bmask,cmap='gray'), plt.title('Otsu')
        #     # plt.show()
            
        #     self.display(octa_nm, octa_adf, octa_frangi, octa_clahe, octa_bmask)


    def kmeans(self):
        print("{0:=^38}".format(" Kmeans "))

    def otsu(self, img):
        print("{0:=^38}".format(" Otsu "))
        img = (img*255).astype(np.uint8)
        _, bmask = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bmask[np.where(bmask==1)]=2
        bmask[np.where(bmask==0)]=1
        bmask[np.where(bmask==2)]=0
        return bmask

    def frangi(self, img, _sigmas=1, _scale_step=0.001, _black_ridges=False):
        print("{0:=^38}".format(" Frangi "))
        img = (img*255).astype(np.uint8)
        return frangi2d(img, sigmas=_sigmas, scale_step=_scale_step, black_ridges=_black_ridges)

    def fuzzy(self):
        print("{0:=^38}".format(" Fuzzy "))

    def rg(self):
        print("{0:=^38}".format(" Region Growing "))

    def adf(self, img, _niter=2, _kappa=90, _gamma=0.1, _voxelspacing=None, _option=3):
        print("{0:=^38}".format(" Anistropy Diffusion Filter "))
        '''
        img : array_like -> Input image (will be cast to numpy.float).
        niter : integer  -> Number of iterations.
        kappa : integer  -> Conduction coefficient, e.g. 20-100. 
                            kappa controls conduction as a function of the gradient. 
                            If kappa is low small intensity gradients are able to block conduction 
                            and hence diffusion across steep edges. 
                            A large value reduces the influence of intensity gradients on conduction.
        gamma : float    -> Controls the speed of diffusion. Pick a value <=.25 for stability.
        voxelspacing : tuple of floats or array_like -> The distance between adjacent pixels in all img.ndim directions
        option : {1, 2, 3}->    Whether to use the Perona Malik diffusion equation No. 1 or No. 2, or Tukey’s biweight function. 
                                Equation 1 favours high contrast edges over low contrast ones, while equation 2 favours wide regions over smaller ones. 
                                See [R9] for details. Equation 3 preserves sharper boundaries than previous formulations 
                                and improves the automatic stopping of the diffusion. See [R10] for details.
        '''
        return anisotropic_diffusion(img, niter=_niter, kappa=_kappa, gamma=_gamma, voxelspacing=_voxelspacing, option=_option)

    def wavelet(self):
        print("{0:=^38}".format(" Wavelet "))

    def curvelet(self):
        print("{0:=^38}".format(" Curvelet "))

    def CLAHE(self, img):
        # Contrast Limited Adaptive Histogram Equalization
        # img : 0~1
        img = (img*255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=100, tileGridSize=(4,4))
        cl1   = clahe.apply(img)
        return np.asarray(cl1)

    def binarymedian(self, img):
        bmi  = sitk.BinaryMedianImageFilter()
        bmi.SetRadius(3)
        bmi.SetForegroundValue(255)
        bmi.SetBackgroundValue(0)
        img = sitk.GetImageFromArray(octa_clahe, isVector=True)
        bmi_img = bmi.Execute(img)
        return sitk.GetArrayFromImage(bmi_img)

    def display(self,nm,adf,frngi,clahe,otsu):
        
        plt.subplots(1,5, figsize=(50,10))
        plt.subplot(151), plt.imshow(nm,     cmap='gray'), plt.axis(False), plt.title('Original')
        plt.subplot(152), plt.imshow(adf,    cmap='gray'), plt.axis(False), plt.title('Anistropic Diffusion')
        plt.subplot(153), plt.imshow(frngi,  cmap='gray'), plt.axis(False), plt.title('Frangi')
        plt.subplot(154), plt.imshow(clahe,  cmap='gray'), plt.axis(False), plt.title('CLAHE')
        plt.subplot(155), plt.imshow(otsu,   cmap='gray'), plt.axis(False), plt.title('Otsu')
        plt.show()

        # plt.subplots(1,5, figsize=(50,10))
        # plt.subplot(151), plt.hist(nm),     plt.title('Original')
        # plt.subplot(152), plt.hist(adf),    plt.title('Anistropic Diffusion')
        # plt.subplot(153), plt.hist(frngi),  plt.title('Frangi')
        # plt.subplot(154), plt.hist(clahe),  plt.title('CLAHE')
        # plt.subplot(155), plt.hist(otsu),   plt.title('Otsu')
        # plt.show()





    # def addHeader(self, nii, dnum):
    #     nib_img= nib.load(os.path.join(self.niidir, nii))
    #     head = nib_img.header
    #     head['dim'][0:6]=[1,400,640,400,1,1]
    #     head['pixdim'][1:4]=[15,3.125,15]
    #     head['xyzt_units']='3'
    #     head['qform_code']='0'
    #     print(head)
    #     c = np.array(nib_img.get_fdata())
    #     nib_img2 = nib.Nifti1Image(c, nib_img.affine, header=head)
    #     head2 = nib_img2.header
    #     head2['dim'][0:6]=[1,400,640,400,1,1] #<---------------Not working ...
    #     head2['pixdim'][1:4]=[15,3.125,15]
    #     head2['xyzt_units']='3'
    #     print(head2)

    #     self.nibname = dnum+'.nii.gz'
    #     self.nibpath = os.path.join(self.niidir, nii)

    #     nib.save(nib_img2, self.nibpath)



        # original __call__
        # self.filter = input("Choice Image Filter:\n(1) KMeans\n(2) Otsu\n(3) Frangi\n(4) Fuzzy\n\
        #                                          \r(5) Region Growing\n(6) Anistropic Diffusion Filter\n\
        #                                          \r(7) Wavelet\n(8) Curvelet : ")
        
        # if   self.filter == '1': self.kmeans()
        # elif self.filter == '2': self.otsu()
        # elif self.filter == '3': self.frangi()
        # elif self.filter == '4': self.fuzzy()
        # elif self.filter == '5': self.rg()
        # elif self.filter == '6': self.adf()
        # elif self.filter == '7': self.wavelet()
        # elif self.filter == '8': self.curvelet()
        # else : return "You need to choose between (1)~(8)"


        # 3d frangi
        # octa_nii_list = sorted(glob.glob(os.path.join(self.niidir,'*')), key=os.path.getctime)
        
        # for nii in octa_nii_list:
        #     self.addHeader(nii, nii.split('.')[0])

        # octa_sitk   = sitk.ReadImage(octa_nii_list[0])
        # octa_array  = sitk.GetArrayFromImage(octa_sitk)

        # octa_frangi1 = frangi3d(octa_array[:,:,:,0], scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500)
        # octa_frangi2 = frangi3d(octa_array[:,:,:,0], scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500)
        # octa_frangi3 = frangi3d(octa_array[:,:,:,0], scale_range=(1, 10), scale_step=2, alpha=0.5, beta=0.5, frangi_c=500)

        
        # plt.subplots(1,3,figsize=(10,30))
        # plt.subplot(131), plt.imshow(octa_frangi[:,300,:], cmap='gray'), plt.axis(False)
        # plt.subplot(132), plt.imshow(octa_frangi[:,350,:], cmap='gray'), plt.axis(False)
        # plt.subplot(133), plt.imshow(octa_frangi[:,400,:], cmap='gray'), plt.axis(False)
        # plt.show() See [R10] for details.
 