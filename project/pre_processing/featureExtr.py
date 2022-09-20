from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd


import cv2 as cv

class FeatureExtractor:

    def __init__(self):
        self.model=VGG16(weights='imagenet', include_top=False, input_shape=(424,424,3))

    def extractFeatures(self,imgpath):
        img=cv.imread(imgpath)
        img_data = np.array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        vgg16_feature = self.model.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        

        return vgg16_feature_np.flatten()

if __name__=='__main__':
    imgPath='./images/training/100335.jpg'
    model=FeatureExtractor()
    model.extractFeatures(imgPath)