#classe specializzata nel flattering di img, ovvero vettorizzazione delle dimensioni hight e width
import cv2 as cv
import numpy as np
class FlatternImage:

    def __init__(self,chanal):
        self.chanal=chanal

    def convertImgRGB(self, image):

        image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
        return image

    def imageToVector(self,image):
        pixel_values = image.reshape((-1, 3))
        # convert to float
        pixel_values = np.float32(pixel_values)
        
        return pixel_values

    def flatterImage(self,image):

        if self.chanal==3:

            RGBimage=self.convertImgRGB(image)
        else:
            RGBimage=image

        flatteredImage=self.imageToVector(RGBimage)
        return flatteredImage

