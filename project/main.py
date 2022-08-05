
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

def printLoadingBar(count,total,suffix):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('PREPROCESSING IMAGE:IN PROGRESS: [%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()  

def createFeaturesAndLabels(X_features):
    features=[]
    labels=[]
    for i in X_features:
        features.append(i.features)
        labels.append(i.label)
        
    outLabels=pd.DataFrame(labels,columns=['target'])
    outFeatures=pd.DataFrame(features)
    return (outFeatures,outLabels)

def visualizeDataPlot(dimensionePlot,featurseNorm):

    if dimensionePlot==3:
        visualize3D(featuresNorm)

    elif dimensionePlot==2:
        visualize2D(featuresNorm)

def visualize2D(featuresNorm):
    pca=PCA(n_components=2)
    principalComponents=pca.fit_transform(featuresNorm)

    #dataframe conententi le due componenti estratte
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
    #final df con cui effettuare la visualizzazione
    
    finalDf = pd.concat([principalDf, labels], axis = 1)
    
    print(finalDf)
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
   
    ax.set_title('3 component PCA', fontsize = 20)
    targets = [0, 1, 2,3,4]
    colors = ['r', 'g', 'b','y']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
               
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

def visualize3D(featuresNorm):
    pca=PCA(n_components=3)
    principalComponents=pca.fit_transform(featuresNorm)

    #dataframe conententi le due componenti estratte
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])
    finalDf = pd.concat([principalDf, labels], axis = 1)

    targets = [0, 1, 2,3,4]
    colors = ['r', 'g', 'b','y']
    

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor('white')
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , finalDf.loc[indicesToKeep, 'principal component 3']          
                , c = color
                , s = 50)

        
    # for loop ends
    ax.set_xlabel("First Principal Component", fontsize=14)
    ax.set_ylabel("Second Principal Component", fontsize=14)
    ax.set_zlabel("Third Principal Component", fontsize=14)

    ax.legend()
    plt.show()

def preprocess(preprocessingMethod,pcaComponents,X_training,Y_training,method,sampleSize):
    #Preprocessamento immagini
    #X_features=pd.DataFrame(columns=['id','fullPath','features','label'])
    X_features=[]
    print("=========================================================")
    print("================PREPROCESSING IMG========================")
    print("=========================================================")
    print("STARTING")
    preprocesso=Preprocessor(preprocessingMethod)     #per il preprocessamento
    nameToImage=NameImage()                                         #per la conversione id img a pattern completo
   
    
    count=0
    #total=len(X_Training)
    if method=='delta':
        total=sampleSize
    
    suffix=''
   
    for i in X_Training:
        
        printLoadingBar(count,total,suffix)
        
        
        img_path= nameToImage.nameToImage(str(i))       #acquisisco il nome completo della immagine
        
        processedImage=preprocesso.preprocess(img_path) #preprocesso l'immagine e ricavo il vettore delle features
        

        label=Y_Training[count]                         #acqusisco il label [etichetta] associata

        
        
        #current=pd.DataFrame({"id":i, "fullPath":img_path,"features":[processedImage],"label":label})

        #X_features=pd.concat([X_features,current],axis = 0)

        #creo una lista degli oggetti 
        current=Pattern(id=i,fullPath=img_path,features=processedImage,label=label)
        X_features.append(current)
        count+=1
        if count==total and method=='delta':
            break

    print("COMPLETED")    
    
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
    print("Metodo di run:", method)
    print("Sample size:",sampleSize)
    
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

    X_features=preprocess(preprocessingMethod=preprocessingMethod,pcaComponents=pcaComponents,X_training=X_Training,Y_training=Y_Training,method=method,sampleSize=sampleSize)

    features_dataFrame,labels=createFeaturesAndLabels(X_features)
    
    #scrittura del risultato del preprocessamento nel csv
    #csvWriter=CSVwriter()                                           #per la scrittura
    #csvWriter.writePatterns(type=preprocessingMethod,pcaApplied=False,data=X_features)
    #csvWriter.readPatterns(type=preprocessingMethod,pcaApplied=False)
    

    #=================================================================
    #====================PCA==========================================
    #abbiamo un dataframe contenetne tutte le info necessarie
    #bisgna normalizzare i valori e applicare la PCA sulle features
    #=================================================================
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    
    
   
    
    

    #standardizzazione delle features (forse meglio da fare direttamente nella fase di preprocessing)
    featuresNorm=pd.DataFrame(StandardScaler().fit_transform(features_dataFrame))
    
    
    #visualizzazione
    if(visualizeData=='True'):
       viewerData=ViewData(dimensionPlot)
       viewerData.visualize(featuresNorm,labels)
    





   
   
    
    
    #===============================2d======================
    
   

    #==================================3d=============================
    



        
        

        
    


                    


        
        
        

        

        
            
        

    
        

        










    
