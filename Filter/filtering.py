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
        
        self.filter = input("Choice Image Filter:\n(1) KMeans\n(2) Otsu\n(3) Frangi\n(4) Fuzzy\n \
                                                 \r(5) Region Growing\n(6) Anistropic Diffusion Filter\n \
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
        frangi()

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

