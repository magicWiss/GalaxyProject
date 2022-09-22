
import cv2 as cv
import numpy as np
from matplotlib import  cm, pyplot as plt


class ThresholdImg:
    def __init__(self):
        
       pass
        


    
    def threshImageOtsu(self,img):
        imgBlurry = cv.medianBlur(img,7)
        th3,ret3 =cv.threshold(imgBlurry,0,4,cv.THRESH_BINARY+cv.THRESH_OTSU)
        #imgBlur=np.multiply(ret3,np.multiply(img,imgBlurry))
        imgBlur=np.multiply(ret3,imgBlurry)
        
        return imgBlur

    def threshImageBinary(self,img):
        
        threshold=np.max(img)*0.6
        
        ret1,th1 = cv.threshold(img,threshold,1,cv.THRESH_BINARY)
        maxrth=np.max(ret1)
        
        imgBlur=np.multiply(th1,img)

       
        
        return imgBlur
        

if __name__=='__main__':
    imgPath='../images/training/101501.jpg'
    img = cv.imread(imgPath,0)
    thres=ThresholdImg()
    img=thres.threshImageOtsu(img)
    










 
