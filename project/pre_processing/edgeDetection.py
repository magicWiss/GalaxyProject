
import cv2 as cv
import numpy as np
from matplotlib import  cm, pyplot as plt
from PIL import Image

class EdgeDetection:
    def __init__(self):
        pass

    def edgeDetect(self,image):

        canny=cv.Canny(image,30,100)
        return canny

    def auto_canny(self, image, sigma=0.33):
	# compute the median of the single channel pixel intensities
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.GaussianBlur(gray, (3, 3), 0)
        v = 0.1*np.max(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv.Canny(image, lower, upper)
        # return the edged image
        return edged


if __name__=='__main__':
    imgPath='./images/training/100008.jpg'
    img = cv.imread(imgPath)
    
    
    
    detector=EdgeDetection()
    edges=detector.auto_canny(img)
    titles=['img','edge']
    images=[img,edges]
    for i in range(2):
        plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    
    
    plt.show()
    