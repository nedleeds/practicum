from genericpath import isfile
import os
import cv2
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
import pandas as pd

class dataLoad():
    def __init__(self, datadir, outdir):
        self.crops = []
        self.octa = {}
        self.DATADIR = datadir
        self.OUTDIR = outdir
        self.width = 0
        self.height = 0

    def __call__(self):
        if self.DATADIR.split('/')[-1]!='label':
            flist = sorted(glob.glob(os.path.join(self.DATADIR, '*')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
            for f in flist:
                img = cv2.imread(os.path.join(f), cv2.IMREAD_GRAYSCALE)
                # center_x, center_y = int(np.shape(img)[0]/2), int(np.shape(img)[1]/2)
                # croped = img[center_x-128:center_x+128, center_y-128:center_y+128]
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
    
class dataCategorize():
    def __init__(self) -> None:
        pass
    def __call__(self, label, nmz_octa):
        basepath = "/root/Share/data/dataset/disease"
        ogpath = "/root/Share/data/dataset/og"
        drpath = os.path.join(basepath, "DR")
        ncpath = os.path.join(basepath, "NORMAL")
        ampath = os.path.join(basepath, "AMD")
        cnpath = os.path.join(basepath, "CNV")
        cspath = os.path.join(basepath, "CSC")
        rvpath = os.path.join(basepath, "RVO")
        otpath = os.path.join(basepath, "OTHERS")

        no, am, dr, cn, cs, rv, ot = {},{},{},{},{},{},{}
        #10217, 10243, 10255 --> is this Normal??
        for i in nmz_octa.keys():
            skipNM = ["10009","10013","10016","10017","10026","10034","10037","10039","10042","10044",
                      "10061","10076","10080","10082","10124","10150","10153","10154","10156","10160",
                      "10172","10174","10185","10191","10207","10213","10214","10215","10217","10231",
                      "10232","10234","10243","10252","10267","10271","10273","10283"]
            skipDR = ["10050",  "10068", "10086", "10135", "10152", "10181", "10184", "10268"]

            k = label[int(i)]
            v = cv2.imread(os.path.join(ogpath, str(i)+".png"), cv2.IMREAD_GRAYSCALE)

            if k=="DR" and i not in skipDR:
                if os.path.isfile(os.path.join(drpath, i+"_DR.png")) : pass
                else : cv2.imwrite(os.path.join(drpath, i+"_DR.png"), v)
                dr[i]=[k]
                dr[i].append(nmz_octa[i])
            elif k=="NORMAL" and i not in skipNM:
                if os.path.isfile(os.path.join(ncpath, i+"_NORMAL.png")) : pass
                else : cv2.imwrite(os.path.join(ncpath, i+"_NORMAL.png"), v)
                no[i]=[k]
                no[i].append(nmz_octa[i])
            elif k=="AMD":
                if os.path.isfile(os.path.join(ampath, i+"_AMD.png")) : pass
                else : cv2.imwrite(os.path.join(ampath, i+"_AMD.png"), v)
                am[i]=[k]
                am[i].append(nmz_octa[i])
            elif k=="CSC":
                if os.path.isfile(os.path.join(cspath, i+"_CSC.png")) : pass
                else : cv2.imwrite(os.path.join(cspath, i+"_CSC.png"), v)
                cs[i]=[k]
                cs[i].append(nmz_octa[i])
            elif k=="CNV":
                if os.path.isfile(os.path.join(cnpath, i+"_CNV.png")) : pass
                else : cv2.imwrite(os.path.join(cnpath, i+"_CNV.png"), v)
                cn[i]=[k]
                cn[i].append(nmz_octa[i])
            elif k=="RVO":
                if os.path.isfile(os.path.join(rvpath, i+"_RVO.png")) : pass
                else : cv2.imwrite(os.path.join(rvpath, i+"_RVO.png"), v)
                rv[i]=[k]
                rv[i].append(nmz_octa[i])
            elif k=="OTHERS":
                if os.path.isfile(os.path.join(otpath, i+"_OTHERS.png")) : pass
                else : cv2.imwrite(os.path.join(otpath, i+"_OTHERS.png"), v)
                ot[i]=[k]
                ot[i].append(nmz_octa[i])
            else: pass
        return no, am, dr, cn, cs, rv, ot
    
class dataMerge():
    def __init__(self) -> None:
        pass
    def __call__(self, *args):
        dataset, dataset2 = {}, {}
        n = len(args)
        class_num = [len(args[x]) for x in range(len(args))]
        for i in range(n):
            for cnt, k in enumerate(args[i]):
                if cnt<class_num[i]//2:
                    dataset[k] = args[i][k]
                else :
                    dataset2[k] = args[i][k]
        dataset.update(dataset2)
        # dataset = sorted(dataset.items(), key=lambda x : x[0])
        return dict(dataset), n
