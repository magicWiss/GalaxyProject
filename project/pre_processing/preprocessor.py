#classe relativa al preprocessamento di una immagine.
#consiste in più metodo differenti di effettuare un preprocessamento
#metodo 1-> crop (160*160) e eliminazione BG mediante threshold otsu
#metodo 2-> crop (160*160)+ eliminazione BG mediante threshold globale
#metodo 2->crop(inteliggente) + eliminazione BG + CNN per feature selction

#in ingresso riceve il path della image, in output restituisce la img preprocesssata

from pre_processing.cropImage import CropImage
from pre_processing.thresholding import ThresholdImg
from pre_processing.edgeDetection import EdgeDetection
from pre_processing.colorAdj import ColorBoost
from pre_processing.featureExtr import FeatureExtractor
import cv2 as cv
from matplotlib import  cm, pyplot as plt
from pre_processing.flatternImage import FlatternImage
import numpy as np
class Preprocessor:

    def __init__(self, method) -> None:
      self.method=method
      if self.method==4:
        self.vgg16=FeatureExtractor()
      

#preprocessamento 1-> cropp img (424*424->160*160)+ threshold con Otsu
#Input-> path imag
#output->scrittura in un file csv delle features


    def preprocess(self, imgPath):
        if self.method==1:
            return self.preprocessOne(imgPath)
        elif self.method==2:
            return self.preprocessTwo(imgPath)
        elif self.method==3:
            return self.preprocess3(imgPath)
        elif self.method==4:
            return self.preprocess4(imgPath)

    def preprocessOne(self, imgPath):
        img = cv.imread(imgPath,0)

        #cropper
        img=np.array(img)
        cropper=CropImage()
        #thresholder
        thresh=ThresholdImg()

        flatter=FlatternImage(chanal=1)
        #flatternImage
        


        #crop image
        croppedImg=np.array(cropper.cropImage(img))
       

        #delete bg
        thresholdImage=np.array(thresh.threshImageOtsu(croppedImg))

       

        #vectorize
        finalImg=np.array(flatter.flatterImage(thresholdImage))

        
        

        return finalImg

    def preprocessTwo(self, imgPath):
        img = cv.imread(imgPath)

        
        #cropper
        img=np.array(img)
        cropper=CropImage()
        #thresholder
        thresh=ThresholdImg()

        flatImageProcessor=FlatternImage(chanal=3)
        
        

        
        #crop image
        croppedImg=np.array(cropper.cropImage(img))
       

        #delete bg
        thresholdImage=np.array(thresh.threshImageBinary(croppedImg))

        

        

        #vettorizzazione dell'img trasformandolo da un 3d array (160*160*3) in un 1 d array (76800)
        finalImg=flatImageProcessor.flatterImage(thresholdImage)
        
        

       
        

        return finalImg

    def preprocess3(self,imgPath):
        img = cv.imread(imgPath)

        
        #cropper
        img=np.array(img)
        cropper=CropImage()
        #thresholder
        thresh=ThresholdImg()

        flatImageProcessor=FlatternImage(chanal=3)
        
        colorBright=ColorBoost()
        

        

        
        #crop image
        croppedImg=np.array(cropper.cropImage(img))
       

        #delete bg
        thresholdImage=np.array(thresh.threshImageBinary(croppedImg))

        #adjustBright
        thresholdImage=np.array(colorBright.boostBright(thresholdImage, method='in'))
       

        edge=EdgeDetection()

        imgEdge=edge.auto_canny(thresholdImage)
        
        
        

        

        

        #vettorizzazione dell'img trasformandolo da un 3d array (160*160*3) in un 1 d array (76800)
        finalImg=flatImageProcessor.flatterImage(imgEdge)
        return finalImg

    def preprocess4(self, imgpath):

        
        features=self.vgg16.extractFeatures(imgpath)
        return features
        

if __name__=='__main__':
    imgPath='./images/training/999875.jpg'
    
    processImg=Preprocessor(3)
    imgProcess, crop, thres=processImg.preprocess(imgPath)
    titles=['img','cropo','thresh','edge']
    images=[cv.imread(imgPath),crop, thres, imgProcess]
    for i in range(4):
        plt.subplot(1,4,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.imshow(imgProcess,cmap='gray')
    
    plt.show()
   





