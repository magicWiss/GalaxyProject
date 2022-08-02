'''
Classe per la principal component analysis.
Riduzione delle dimensioni da 160*160 in 125 features.


'''
from sklearn import datasets
from sklearn.decomposition import PCA
import cv2 as cv
import pandas as pd

class PrincipalComonentAnalysis:
    def __init__(self,components):
        self.components=components
        self.pca=PCA()

    #questa è la riduzione di dimensionalità avente un solo canale
    def reduceComponentsSingleChannel(self,dataSet):
        return self.pca.fit_transform(dataSet)



    #questa è la riduzione di dimensionalità in presenza di 3 canali RGB 
    def reduceComponentsThreeChannel(self,dataSet):
        blue,green,red = cv.split(dataSet)

        #normalizzazione delle features
        blue_df=pd.DataFrame(blue)
        blue_df=blue_df/255
        red_df=pd.DataFrame(red)
        red_df=red_df/255
        green_df=pd.DataFrame(green)
        green_df=green_df/255

        #fit e trasform delle componenti
        self.pca.fit(blue_df)
        
        trans_pca_b = self.pca.transform(blue_df)

        b_arr = self.pca.inverse_transform(trans_pca_b)
        
        print(f"Blue Channel : {sum(self.pca.explained_variance_ratio_)}")
        
        self.pca.fit(green_df)
        trans_pca_g = self.pca.transform(green_df)
        g_arr = self.pca.inverse_transform(trans_pca_g)
        
        print(f"Green Channel: {sum(self.pca.explained_variance_ratio_)}")


        self.pca.fit(red_df)
        trans_pca_r = self.pca.transform(red_df)
        
        print(f"Red Channel  : {sum(self.pca.explained_variance_ratio_)}")

        #ricostrizuine della img
        
        r_arr = self.pca.inverse_transform(trans_pca_r)

        img_reduced= (cv.merge((b_arr, g_arr, r_arr)))

        return img_reduced

    def inverted_fit_trasform(self, dataSet):
        return self.pca.inverse_transform(dataSet)
