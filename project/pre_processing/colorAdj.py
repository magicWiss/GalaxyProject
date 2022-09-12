import cv2 as cv
from pylab import array, plot, show, axis, arange, figure, uint8 
from matplotlib import  cm, pyplot as plt

class ColorBoost:

    def __init__(self) -> None:
        pass
    


    def increase(self, image, x, phi, theta, maxIntensity):
        newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
        newImage0 = array(newImage0,dtype=uint8)
        return newImage0


    def decrease(self, image, y, phi, theta, maxIntensity):
        newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
        newImage1 = array(newImage1,dtype=uint8)
        return newImage1

    def boostBright(self, img, method):

        maxinten=255.0
        x=arange(maxinten)


        phi=2
        theta=2

        if method=='in':
            return self.increase( img, x, phi, theta, maxinten)
        elif method=='de':
            y = (maxinten/phi)*(x/(maxinten/theta))**0.5
            return self.decrease(img,y,phi,theta,maxinten )
        

if __name__=='__main__':        
        imgPath='./images/training/100335.jpg'
        img = cv.imread(imgPath,0)
       
        
        color=ColorBoost()
        imgBright=color.boostBright(img, method='in')
        plt.imshow(imgBright)
        
        plt.show()