import cv2 as cv
import numpy as np
from matplotlib import  cm, pyplot as plt
'''
Per chiara.

Utilizza il main per invocare il tuo metodo e la classe CropImage per definire il metodo di
riduzione della image (cropImage)
Quest'ultimo dovrà restituire una image
Per stampare la imge inserisci questi comandi nel codice


plt.imshow(imgBlur,cmap='gray')
plt.show()

Stampa a video in scala grigia del risultato

'''
class CropImage:
    def __init__(self):
        pass

    def cropImage(self,image):

        pass




if __name__=='__main__':
    imgPath='../images/training/100335.jpg'
    img = cv.imread(imgPath)
    cropper=CropImage()
    img=cropper.cropImage(img)