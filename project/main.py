
import os

import pandas as pd
import sklearn as skl
import numpy as np
import json


import sys
from matplotlib import  cm, pyplot as plt
from keras.applications.vgg16 import VGG16
from pre_processing.featureExtr import FeatureExtractor
from pre_processing.preprocessor import Preprocessor
from pre_processing.PCA import PrincipalComponentAnalysis
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
    pcaComponents=data['pcaNComponents']

    #method
    method=data['method']               #full->tutti i dati, delta->subset

    #sampleSize
    sampleSize=data['sampleSize']

    #visualizzazione dati
    visualizeData=data['visualizeData']     #true visualizzi, false non visualizzi

    #tipo di visualizzazione
    dimensionPlot=data['dimensionPlot']     #3d o 2d

    #utilizzo VGG16 per features extraction
    use_vgg16=data['VGG16']


    
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
    
    
   #Da fare un multiBar- plot
    if(visualizeData=='True'):
        stats=all_data['label'].value_counts()
        training_stats=training_data['label'].value_counts(normalize=True)
        training_stats.plot.bar()
        test_stats=test_data['label'].value_counts()
        plt.show()
    

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
    X_Test,Y_Test=createFeaturesAndLabels(X_featuresTest)   #da rivedere

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

    #======================================================================
    #===========================VGG16 Features extraction==================
    #======================================================================
    if use_vgg16==True:
        if preprocessingMethod==1 or preprocessingMethod==3:
            size=(160,160,1)
        else:
            size=(160,160,3)
        VGG_model=VGG16(weights='imagenet',include_top=False, inputSize=size)

        

        #estrazione dellefeatures per Training
        features_extractor=VGG_model.predict(X_TrainingNorm)
        X_TrainingNorm=features_extractor.reshape(features_extractor.shape[0],-1)

        #estrazione features per test
        features_extractor=VGG_model.predict(X_TestNorm)
        X_TestNorm=features_extractor.reshape(features_extractor.shape[0],-1)


    #=======================================================================
    #=========================PLOT DEI DATI=================================
    #=======================================================================
    
    #visualizzazione
    if(visualizeData=='True'):
       viewerData=ViewData(dimensionPlot,'LDA')
       viewerData.visualize(X_TrainingNorm,Y_Training)

    #=========================================================================
    #=========================PCA or LDA==========================================
    #=======================================================================
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    #tramite il file json prendiamo il parametro che ci dice quante componenti vogliamo nella pca
    methodRedComp=data["redComMet"]

    if (methodRedComp=="LDA" or methodRedComp=='lda'):
        my_LDA=LDA(solver='svd')
        X_Train_Rid=pd.DataFrame(my_LDA.fit_transform(X_TrainingNorm.fillna(0),Y_Training)).fillna(0)   
        X_Test_Rid=pd.DataFrame(my_LDA.transform(X_TestNorm.fillna(0))).fillna(0)
    
    else:
        my_pca= PrincipalComponentAnalysis(pcaComponents)
   
    #applicazione della pca ai set già normalizzati
        X_Train_Rid=pd.DataFrame(my_pca.pcaFunction(X_TrainingNorm.fillna(0))).fillna(0) 
        components=my_pca.pca.n_components_
        #test_pca=my_pca = PrincipalComponentAnalysis(components)
        X_Test_Rid=pd.DataFrame(my_pca.pcaFunctionTest(X_TestNorm.fillna(0))).fillna(0)
    
    #stampa di tutti i parametri della pca
    #my_pca.printParam()
    del X_TrainingNorm
    del X_TestNorm
    if(visualizeData=='True'):
            viewerData=ViewData(dimensionPlot,'LDA')
            viewerData.visualize(X_Train_Rid,Y_Training)        
        


    #=========================================================================
    #=========================MODEL==========================================
    #=======================================================================

    #lettura del modello da utilizzare per il run corrente dal file di parametri
    ML_model=data['model']

    #passiamo i modelli normalizzati e ridimensionati tramite pca
    modelRouter=ModelRouter(ML_model)
    print("shape trainig data: ", X_Train_Rid.shape)
    print("shape trainig data: ", Y_Training.shape)
    print("shape test data: ", X_Test_Rid.shape)
    print("shape test data: ", Y_Test.shape)
    modelRouter.trainModel(X_Train_Rid,Y_Training,X_Test_Rid,Y_Test)     #probabilmente qui dovremmo passare anche per i modelli supervisionati anche X_test e Y_test
    








   
   
    
    
    
    



        
        

        
    


                    


        
        
        

        

        
            
        

    
        

        










    
