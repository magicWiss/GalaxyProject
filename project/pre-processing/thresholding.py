
import cv2 as cv
import numpy as np
from matplotlib import  cm, pyplot as plt


class ThresholdImg:
    def __init__(self):
        
       pass
        


    
    def threshImageOtsu(self,img):
        img = cv.medianBlur(img,5)
        th3,ret3 =cv.threshold(img,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
        imgBlur=np.multiply(ret3,img)
        plt.imshow(imgBlur,cmap="gray")
        plt.show()
        return imgBlur

    def threshImageBinary(self,img):
        
        threshold=np.max(img)*0.1
        
        ret1,th1 = cv.threshold(img,threshold,1,cv.THRESH_BINARY)
        maxrth=np.max(ret1)
        
        imgBlur=np.multiply(th1,img)
       
        plt.imshow(imgBlur)
        plt.show()
        return imgBlur
        

if __name__=='__main__':
    imgPath='../images/training/101501.jpg'
    img = cv.imread(imgPath)
    thres=ThresholdImg()
    img=thres.threshImageBinary(img)
    










 
