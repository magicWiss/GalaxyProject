
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
    preprocesso=Preprocessor(preprocessingMethod,pcaComponents)
    nameToImage=NameImage()
    
    count=0
    #total=len(X_Training)
    total=10
    
    suffix=''
    for i in X_Training:
        
        printLoadingBar(count,total,suffix)
        
        print("stats variance img: ", str(i))
        img_path= nameToImage.nameToImage(str(i))
        
        processedImage=preprocesso.preprocess(img_path)
        X_features.append(processedImage)
        count+=1
        if count==total:
            break
        
    
    print("\n\nCOMPLETED")


    '''#da modificare
    feat=np.array(X_features)
    feat=feat.reshape(-1,2)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    model_kMeans=KMeans(n_clusters=5, init='k-means++')

    model_kMeans.fit(feat)

    file1 = open("labels.txt", "w")

    
    with open('project/training_labels.csv', 'r') as my_file:
        file_csv = reader(my_file)
        head = next(file_csv)

        # check if the file is empty or not
        if head is not None:
            # Iterate over each row
            conta=0
            for i in file_csv:
                id=i[0]
                label=i[1]
                predict=model_kMeans.labels_[conta]
                conta+=1
                file1.write("id->"+str(id)+"   predicted->"+str(predict)+"   true->"+str(label)+ str(predict==label)+"\n")
                if conta==len(model_kMeans.labels_):
                    break

    file1.close()

                


    #score = silhouette_score(feat, model_kMeans.labels_, metric='euclidean')
    #
    # Print the score
    #
    #print('Silhouetter Score: %.3f' % score)
    
  '''
    
    

    

    
        
    

    #da aggiungere una normalizzazione delle features 

    

    










    #lavoro su training_labels
    
    #per ogni image nel trainging
        #preprocessing
        #pca
        
    #applicazione di un modello
    #validazione
    
    #grafica

    #test
