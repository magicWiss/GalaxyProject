'''
Classe per la principal component analysis.
Riduzione delle dimensioni da 160*160 in 125 features.


'''
from sklearn import datasets
from sklearn.decomposition import PCA
import cv2 as cv
import pandas as pd
import numpy as np

class PrincipalComonentAnalysis:
    def __init__(self,components):
        self.components=components
        self.pca=PCA()

    


    #questa è la riduzione di dimensionalità in presenza di 3 canali RGB 
    def reduceComponents(self,dataSet):
        
        #fit e trasform delle componenti
        self.pca.fit(dataSet)
        trans_pca = np.array(self.pca.transform(dataSet))     
        
        print(f"Blue Channel : {sum(self.pca.explained_variance_ratio_)}")      
        
        #print(f"Red Channel  : {sum(self.pca.explained_variance_ratio_)}")

        #ricostrizuine della img
        return trans_pca

    def inverted_fit_trasform(self, dataSet):
        return self.pca.inverse_transform(dataSet)
