
import os

import pandas as pd
import sklearn as skl
import numpy as np
import json
import time
import sys
from matplotlib import  cm, pyplot as plt

from  pre_processing.preprocessor import Preprocessor
from pre_processing.PCA import PrincipalComonentAnalysis
from csv import reader
from pre_processing.nameToImage import NameImage
from classes.pattern import Pattern
from classes.csvWriter import CSVwriter

def printLoadingBar(count,total,suffix):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('PREPROCESSING IMAGE:IN PROGRESS: [%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()  


def preprocess(preprocessingMethod,pcaComponents,X_training,Y_training):
    #Preprocessamento immagini
    X_features=[]
    print("=========================================================")
    print("================PREPROCESSING IMG========================")
    print("=========================================================")
    print("STARTING")
    preprocesso=Preprocessor(preprocessingMethod,pcaComponents)     #per il preprocessamento
    nameToImage=NameImage()                                         #per la conversione id img a pattern completo
   
    
    count=0
    total=len(X_Training)
    
    
    suffix=''
    for i in X_Training:
        
        printLoadingBar(count,total,suffix)
        
        
        img_path= nameToImage.nameToImage(str(i))       #acquisisco il nome completo della immagine
        
        processedImage=preprocesso.preprocess(img_path) #preprocesso l'immagine e ricavo il vettore delle features

        label=Y_Training[count]                         #acqusisco il label [etichetta] associata

        pattern=Pattern(id=str(i),fullPath=img_path,features=processedImage, label=label)


        X_features.append(pattern)
        count+=1
        
        
    
    print("\n\nPREPROCESSING COMPLETED")

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
    X_Training=np.array(training_data['imageId'])
    #training labels
    Y_Training=np.array(training_data['label'])

    #Test id
    X_Test=np.array(test_data['imageId'])
    
    Y_Test=np.array(test_data['label'])
   
    #=================================================================
    #====================PREPROCESSAMENTO IMG=========================
    #=================================================================

    preprocess(preprocessingMethod=preprocessingMethod,pcaComponents=pcaComponents,X_training=X_Training,Y_training=Y_Training)


        #scrittura del risultato del preprocessamento nel csv
    #csvWriter=CSVwriter()                                           #per la scrittura
    #csvWriter.writePatterns(type=preprocessingMethod,pcaApplied=False,data=X_features)
    #csvWriter.readPatterns(type=preprocessingMethod,pcaApplied=False)
    

    
    

                


    
    
    

    

    
        
    

   
    

    










    #lavoro su training_labels
    
    #per ogni image nel trainging
        #preprocessing
        #pca
        
    #applicazione di un modello
    #validazione
    
    #grafica

    #test
