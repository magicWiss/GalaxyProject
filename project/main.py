
import os

import pandas as pd
import sklearn as skl
import numpy as np
import json
import time
import sys

from  pre_processing.preprocessor import Preprocessor
from pre_processing.PCA import PrincipalComonentAnalysis

from pre_processing.nameToImage import NameImage

def printLoadingBar(count,total,suffix):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('PREPROCESSING IMAGE:IN PROGRESS: [%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()  

if __name__=="__main__":
    
    #=====================================================
    #========LETTURA PARAMETRI DI PREPROCESSINF===========
    #=====================================================
    with open('project/parameterFile.json', 'r') as f:
        data = json.load(f)                         #è un dizionario contentente tutti i parametri utilizzati

    
    #metriche training e test
    perc_training=data['training_perc']

    #metodo di preprocessamento
    preprocessingMethod=data['preProcessingMethod']

    #numero componenti nella pca
    pcaComponents=data['PCAComponents']
    
    #========================================================
    #=============FINE LETTURA PARAMETRI ======
    #========================================================

    #========================================================
    #=============CREAZIONE DATASET==========================
    #========================================================

    #creazione del training set e test set a partire da file trainig_labels.csV
    all_data=pd.read_csv('project/training_labels.csv')
    #shuffle del dataframe per evitare qualsisai dipendenza nella sequenza di immagini
    

    
    #divisione Training e Test-set
    numberOfRowsTraining=round(all_data.shape[0]*perc_training)
    
  
    training_data=all_data.iloc[:numberOfRowsTraining, :]
    test_data=all_data.iloc[numberOfRowsTraining:, :]

    #check se ci sono righe comuni
    

    print("Dimensione dataset totale:",all_data.shape[0])
    print("Dimensione dataset training:",training_data.shape[0])
    print("Dimensione dataset di test:",test_data.shape[0])
    
    #=========================================================
    #===============FINE CREAZIONE DATASET====================
    #=========================================================
    

    #=============================================================
    #==============SUDDIVISIONE TRAINING E TEST SET===============
    #=============================================================
    #Training id
    X_Training=training_data['imageId']
    #training labels
    Y_Training=training_data['label']

    #Test id
    X_Test=test_data['imageId']
    
    Y_Test=test_data['label']
   
    #=================================================================
    #====================PREPROCESSAMENTO IMG=========================
    #=================================================================

    #Preprocessamento immagini
    X_features=[]
    print("=========================================================")
    print("================PREPROCESSING IMG========================")
    print("=========================================================")
    print("STARTING")
    preprocesso=Preprocessor(preprocessingMethod)
    nameToImage=NameImage()
    
    count=0
    total=len(X_Training)
    suffix=''
    for i in X_Training:
        
        printLoadingBar(count,total,suffix)
        
        img_path= nameToImage.nameToImage(str(i))
        
        processedImage=preprocesso.preprocess(img_path)
        X_features.append(processedImage)
        count+=1
    
    print("\n\nCOMPLETED")
  
    X_features=np.array(X_features)
    print(X_features)

    

    










    #lavoro su training_labels
    
    #per ogni image nel trainging
        #preprocessing
        #pca
        
    #applicazione di un modello
    #validazione
    
    #grafica

    #test
