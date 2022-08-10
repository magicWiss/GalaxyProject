'''
Classe per la principal component analysis.
Riduzione delle dimensioni da 160*160 in 125 features.


'''
from sklearn.decomposition import PCA
import cv2 as cv
import pandas as pd
import numpy as np

class PrincipalComponentAnalysis:

    def __init__(self, n_comp) :
        self.pca=PCA(n_components=n_comp)

    
    def pcaFunction(self,X_set):
        #il training e test set sono già stati normalizzati nel main tramite Scaler
        #Adattare il modello con X e applicare la riduzione della dimensionalità su X.
        
        set_rid=self.pca.fit_transform(X_set)
       
        return set_rid

    
    
    def printParam(self):
        #print('Covarianza dei dati con il modello generativo:')
        #print(self.pca.get_covariance.__str__)
        #print('Parametri:' + self.pca.get_params.__str__)
        #print('Matrice di precisione dei dati con il modello generativo:' + self.pca.get_precision.__str__)
        #print('Quantità di varianza spiegata da ciascuno dei componenti selezionati:' + self.pca.explained_variance_ratio_)
        print('Numero di componenti selezionate nella pca: ', self.pca.n_components_)
        print('Numero di feature selezionate dalla pca: ', self.pca.n_features_in_)
        print('La covarianza del rumore stimata dalla pca: ', self.pca.noise_variance_)

   
   
   
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
