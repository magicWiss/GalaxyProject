#classe specializzata nel flattering di img, ovvero vettorizzazione delle dimensioni hight e width
import cv2 as cv
import numpy as np
class FlatternImage:

    def __init__(self,chanal):
        self.chanal=chanal
        

   
    def flatterImage(self,image):

        return image.ravel()

    
    #utile per mostarare le img
    def unflatterImage(self,img):

        if self.chanal==3:
            return img.reshape((160,160,3))

        if self.chanal==1:
            return img.reshape((160,160))

     

