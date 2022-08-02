#classe relativa al preprocessamento di una immagine.
#consiste in più metodo differenti di effettuare un preprocessamento
#metodo 1-> crop (160*160) e eliminazione BG mediante threshold otsu
#metodo 2-> crop (160*160)+ eliminazione BG mediante threshold globale
#metodo 2->crop(inteliggente) + eliminazione BG + CNN per feature selction

#in ingresso riceve il path della image, in output restituisce la img preprocesssata

from pre_processing.cropImage import CropImage
from pre_processing.thresholding import ThresholdImg
from pre_processing.PCA import PrincipalComonentAnalysis
import cv2 as cv
from matplotlib import  cm, pyplot as plt
from pre_processing.flatternImage import FlatternImage
import numpy as np
class Preprocessor:

    def __init__(self, method,pcaComponent) -> None:
      self.method=method
      self.pcaComponent=pcaComponent

#preprocessamento 1-> cropp img (424*424->160*160)+ threshold con Otsu
#Input-> path imag
#output->img preprocessata


    def preprocess(self, imgPath):
        if self.method==1:
            return self.preprocessOne(imgPath)
        elif self.method==2:
            return self.preprocessTwo(imgPath)


    def preprocessOne(self, imgPath):
        img = cv.imread(imgPath,0)

        #cropper
        img=np.array(img)
        cropper=CropImage()
        #thresholder
        thresh=ThresholdImg()

        flatter=FlatternImage(chanal=1)
        #flatternImage
        

        pca=PrincipalComonentAnalysis(self.pcaComponent)

        #crop image
        croppedImg=np.array(cropper.cropImage(img))
        print("Dimensione dopo crop:", croppedImg.shape)

        #delete bg
        thresholdImage=np.array(thresh.threshImageOtsu(croppedImg))

        print("Dimensioni pre flattering:",thresholdImage.shape)

        #vectorize
        flattedImage=np.array(flatter.flatterImage(thresholdImage))

        print("Dimensione dopo flattering:",flattedImage.shape)

        finalImg=np.array(pca.reduceComponents(flattedImage))

        print("Dimensione finale:",finalImg.shape)
        

        return finalImg

    def preprocessTwo(self, imgPath):
        img = cv.imread(imgPath)

        
        #cropper
        img=np.array(img)
        cropper=CropImage()
        #thresholder
        thresh=ThresholdImg()

        flatter=FlatternImage(chanal=3)
        #flatternImage
        

        pca=PrincipalComonentAnalysis(self.pcaComponent)

        #crop image
        croppedImg=np.array(cropper.cropImage(img))
        print("Dimensione dopo crop:", croppedImg.shape)

        #delete bg
        thresholdImage=np.array(thresh.threshImageBinary(croppedImg))

        print("Dimensioni pre flattering:",thresholdImage.shape)

        #vectorize
        flattedImage=np.array(flatter.flatterImage(thresholdImage))

        print("Dimensione dopo flattering:",flattedImage.shape)

        finalImg=np.array(pca.reduceComponents(flattedImage))

        print("Dimensione finale:",finalImg.shape)
        

        return finalImg



'''if __name__=='__main__':
    imgPath='../images/training/101501.jpg'
 
    preProc=Preprocessor()
    img=preProc.preprocessTwo(imgPath)

    plt.imshow(img)
    plt.show()
'''

