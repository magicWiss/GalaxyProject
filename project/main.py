
import os

import pandas as pd
import sklearn as skl
import numpy as np
import json

import sys
from matplotlib import  cm, pyplot as plt

from  pre_processing.preprocessor import Preprocessor
from pre_processing.PCA import PrincipalComonentAnalysis
from csv import reader
from pre_processing.nameToImage import NameImage
from classes.pattern import Pattern
from classes.csvWriter import CSVwriter
from visualizzatore.viewData import ViewData
from models.router import ModelRouter

def printLoadingBar(count,total,suffix):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('IN PROGRESS: [%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()  

def createFeaturesAndLabels(X_features):
    features=[]
    labels=[]
    total=len(X_features)
    count=0
    suffix=''
    for i in X_features:
        features.append(i.features)
        labels.append(i.label)
        printLoadingBar(count,total,suffix)
        count+=1
        
    outLabels=pd.DataFrame(labels,columns=['target'])
    outFeatures=pd.DataFrame(features)
    return (outFeatures,outLabels)

def preprocess(preprocessingMethod,X_Data,Y_Data,method,sampleSize):
    #Preprocessamento immagini
    #X_features=pd.DataFrame(columns=['id','fullPath','features','label'])
    X_features=[]
   
   
    preprocesso=Preprocessor(preprocessingMethod)     #per il preprocessamento
    nameToImage=NameImage()                                         #per la conversione id img a pattern completo
   
    
    count=0
    
    if method=='delta':
        total=int(sampleSize*len(X_Data))                #il totale dei dati da usare in fase di delta
        
    
    suffix=''
   
    for i in X_Data:
        
        printLoadingBar(count,total,suffix)
        
        
        img_path= nameToImage.nameToImage(str(i))       #acquisisco il nome completo della immagine
        
        processedImage=preprocesso.preprocess(img_path) #preprocesso l'immagine e ricavo il vettore delle features
        

        label=Y_Data[count]                         #acqusisco il label [etichetta] associata

        
        

        #creo una lista degli oggetti 
        current=Pattern(id=i,fullPath=img_path,features=processedImage,label=label)
        X_features.append(current)
        
        if count==total and method=='delta':
            break

        count+=1

      
    
    return X_features
    

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

    #method
    method=data['method']               #full->tutti i dati, delta->subset

    #sampleSize
    sampleSize=data['sampleSize']

    #visualizzazione dati
    visualizeData=data['visualizeData']     #true visualizzi, false non visualizzi

    #tipo di visualizzazione
    dimensionPlot=data['dimensionPlot']     #3d o 2d


    
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
    X_TrainingImgs=np.array(training_data['imageId'])
    #training labels
    Y_TrainingImgs=np.array(training_data['label'])

    #Test id
    X_TestImgs=np.array(test_data['imageId'])
    
    Y_TestImgs=np.array(test_data['label'])

    print("Metodo di run:", method)
    print("Sample training size:",int(sampleSize*len(X_TrainingImgs)))
    print("Sample test size:",int(sampleSize*len(X_TestImgs)))
    
   
    #=================================================================
    #====================PREPROCESSAMENTO IMG=========================
    #=================================================================
    #TRAINING
    print("\nPreprocessing Training data")
    X_featuresTraining=preprocess(preprocessingMethod=preprocessingMethod,X_Data=X_TrainingImgs,Y_Data=Y_TrainingImgs,method=method,sampleSize=sampleSize)
    print("\nExtracting Training features")
    X_Training,Y_Training=createFeaturesAndLabels(X_featuresTraining)


    del X_TrainingImgs
    del Y_TrainingImgs
    del X_featuresTraining
    #preprocessamento del test set

    #TEST
    print("\nPreprocessing Test data")
    X_featuresTest=preprocess(preprocessingMethod=preprocessingMethod,X_Data=X_TestImgs,Y_Data=Y_TestImgs,method=method,sampleSize=sampleSize)
    print("\nExtracting Test features")
    X_Test,Y_Test=createFeaturesAndLabels(X_featuresTest)

    del X_TestImgs
    del Y_TestImgs
    del X_featuresTest
    
    #scrittura del risultato del preprocessamento nel csv
    #csvWriter=CSVwriter()                                           #per la scrittura
    #csvWriter.writePatterns(type=preprocessingMethod,pcaApplied=False,data=X_features)
    #csvWriter.readPatterns(type=preprocessingMethod,pcaApplied=False)
    

    
    
    from sklearn.preprocessing import StandardScaler

    
    
    #=======================================================================
    #=====================Normalizzazione dei dati==========================
    #=======================================================================
    #standardizzazione delle features di training
    X_TrainingNorm=pd.DataFrame(StandardScaler().fit_transform(X_Training))
    del X_Training

    #standardizzazione delle features di test
    X_TestNorm=pd.DataFrame(StandardScaler().fit_transform(X_Test))
    del X_Test



    #=======================================================================
    #=========================PLOT DEI DATI=================================
    #=======================================================================
    
    #visualizzazione
    if(visualizeData=='True'):
       viewerData=ViewData(dimensionPlot)
       viewerData.visualize(X_TrainingNorm,Y_Training)




    
    #=========================================================================
    #=========================MODEL==========================================
    #=======================================================================

    #lettura del modello da utilizzare per il run corrente dal file di parametri
    ML_model=data['model']

    modelRouter=ModelRouter(ML_model)
    modelRouter.trainModel(X_TrainingNorm,Y_Training,X_TestNorm,Y_Test)     #probabilmente qui dovremmo passare anche per i modelli supervisionati anche X_test e Y_test
    








   
   
    
    
    
    



        
        

        
    


                    


        
        
        

        

        
            
        

    
        

        










    
