
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
        print("\nAddestramento di un modello KNN")
        knn = KNeighborsClassifier(n_neighbors=1)
       
        knn.fit(X_Train,Y_Train)
        

        #predizione del modello
        pred = knn.predict(X_Test)

       
        print(confusion_matrix(Y_Test,pred))
        print(classification_report(Y_Test,pred))
        
        #Scelta del valore di k: Per ogni valore di k chiameremo classificatore KNN e quindi sceglieremo il valore di k che ha il minor tasso di errore
        error_rate = []
        error4k={}
        max_k=20

        

        for i in range(1,max_k+1):
    
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_Train,np.array(Y_Train))

            #CROSS VALIDATION
            CV = CrossValidation(i)
            pred_i = CV.kFoldCV(X_Test, np.array(Y_Test), knn.predict(X_Test))
            
            #confronto
            Y_Test=np.array(Y_Test)
            rate=np.mean(pred_i!=Y_Test)
            error_rate.append(rate)
            error4k[i]=np.mean(rate)

        #print("Cross Validation Scores: ", pred_i.scores)
        #print("Average CV Score: ", pred_i.scores.mean())
        #print("Number of CV Scores used in Average: ", pred_i.len(scores))  

        #Tracciamo un grafico a linee del tasso di errore
        print("\nTracciamo un grafico a linee del tasso di errore")
        plt.figure(figsize=(10,6))
        plt.plot(range(1,max_k+1),error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.show()


        #Ora dal grafico dovremo vedere qual è la zona con errore minore e quindi impostare il K noi manualmente e vedere la differenza con il K=1
        #iniziale preso come esempio
        #altrimenti dovremo fare una selezione del minimo dell' Error rate e memorizzare in una variabile il suo K associato così da metterla qui e controllare le differenze
        
        mini=min(error4k,key=error4k.get)
        knn = KNeighborsClassifier(n_neighbors=mini,algorithm='kd_tree')

        knn.fit(X_Train,Y_Train)
        
        
        #addestriamo con il test set
        pred = knn.predict(X_Test)


        print('WITH K=',mini)
        print('\n')
        print(confusion_matrix(Y_Test,pred))
        print('\n')
        print(classification_report(Y_Test,pred))









        #qui si implementa il metodo per l'addesteamento del modello
        #si inseriscono qui le operazioni di:
        #PCA e scelta del numero di componenti ottimali
        #cross-validation per la scelta del numero di k ottimi per il modello
        #stampa del risultato dell'addestramento
        #model=KNeighborsClassifier()
        #model.fit(X_Train,Y_Train)
        #predicted_test=model.predict(X_Test)
        #acc=accuracy_score(Y_Test,predicted_test)

        #print("KNN Accuricy:",acc)
        #print(confusion_matrix(Y_Test, predicted_test))
        #print(classification_report(Y_Test, predicted_test))


        