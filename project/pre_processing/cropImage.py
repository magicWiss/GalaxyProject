
import cv2 as cv
import numpy as np
from matplotlib import  cm, pyplot as plt
from PIL import Image
'''

Cropping della img.
La image è 424*424*3 3 canali rgb.
L'immagine croppata è la trasformazione di una imagw 424*424 in una 160*160
Si devono prendere i 160 pixel più rilevanti ovvero quelli centrali.


Per chiara.

Utilizza il main per invocare il tuo metodo e la classe CropImage per definire il metodo di
riduzione della image (cropImage)
Quest'ultimo dovrà restituire una image croppata nel centro ovvero.

424/2= 212 (centro)
ci spostiamo di 160/2 a sx e dx
limiti
left,top->212-160/2=132
right,buttom->212+160/2=292

Questo implca che in matriciale abbiamo:
dalla riga/colonna 132 alla righa/colonna 292


'''
class CropImage:
    def __init__(self):
        self.left=132
        self.right=292
        self.top=132
        self.bottom=292

    def cropImage(self,image):

        imageCropp=image[self.left:self.right,self.top:self.bottom] #inizio Riga:fine righa, iniziocolonna,fineColonna
        return imageCropp




if __name__=='__main__':
    imgPath='/images/training/100335.jpg'
    img = cv.imread(imgPath,0)
    arr=np.array(img)
    print("size->",(arr.shape))
    cropper=CropImage()
    imgcropped=cropper.cropImage(img)
    plt.imshow(imgcropped,cmap='gray')
    
    plt.show()
