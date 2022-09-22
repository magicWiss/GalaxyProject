
from statistics import mean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from pre_processing.crossValid import CrossValidation
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class KNN:

    def __init__(self):

        pass

    def predict(self, X_Train,Y_Train, X_Test, Y_Test):        
        #Scelta del valore di k: Per ogni valore di k chiameremo classificatore KNN e quindi sceglieremo il valore di k che ha il minor tasso di errore
        error_rate = []
        max_k=20
        best_meanError=np.inf
        for i in range(1,max_k+1):
    
            knn = KNeighborsClassifier(n_neighbors=i,weights='distance',metric='minkowski')
            knn.fit(X_Train,np.array(Y_Train))

            #CROSS VALIDATION
            from sklearn.model_selection import cross_val_score
            #train model with cv of 5 
            cv_scores = cross_val_score(knn, X_Train, np.ravel(Y_Train), cv=5)
            #print each cv score (accuracy) and average them
            
            print("Errore per ogni k-blocci:",cv_scores)
            mean_error=np.mean(cv_scores)
            print("Errore medio con k=",i,"->",mean_error)
            error_rate.insert(i,mean_error)
            if (mean_error<=best_meanError):
                best_meanError=mean_error 
                    


        #Tracciamo un grafico a linee del tasso di errore
        print("\nTracciamo un grafico a linee del tasso di errore")
        plt.figure(figsize=(10,6))
        plt.plot(range(1,max_k+1,1),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Errore medio K-fold')
        plt.show()


        #Ora dal grafico dovremo vedere qual è la zona con errore minore e quindi impostare il K noi manualmente e vedere la differenza con il K=1
        #iniziale preso come esempio
        #altrimenti dovremo fare una selezione del minimo dell' Error rate e memorizzare in una variabile il suo K associato così da metterla qui e controllare le differenze
        
        minimumK=error_rate.index(min(error_rate))  
        
        knn = KNeighborsClassifier(n_neighbors=minimumK+1,weights='distance',metric='minkowski')

        knn.fit(X_Train,np.ravel(Y_Train))
        
        
        #addestriamo con il test set
        pred = knn.predict(X_Test)


        print('Accuracy con k=',minimumK+1)
        print(accuracy_score(Y_Test, pred))
        print('\n')
        print("Matrice di confusione\n",confusion_matrix(Y_Test,pred))
        print('\n')
        print(classification_report(Y_Test,pred))









        


        