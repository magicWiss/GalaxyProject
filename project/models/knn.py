
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
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
        print(X_Test)
        print(X_Train)

        #predizione del modello
        pred = knn.predict(X_Test)

        print("\nValutiamo il nostro modello KNN")
        print(confusion_matrix(Y_Test,pred))
        print(classification_report(Y_Test,pred))
        
        #Scelta del valore di k: Per ogni valore di k chiameremo classificatore KNN e quindi sceglieremo il valore di k che ha il minor tasso di errore
        error_rate = []
        max_k=20

        for i in range(1,max_k+1):
    
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_Train,np.array(Y_Train))
            pred_i = knn.predict(X_Test)
            Y_Test=np.array(Y_Test)
            error_rate.append(np.mean(pred_i != Y_Test))



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
        
        knn = KNeighborsClassifier(n_neighbors=23)

        knn.fit(X_Train,Y_Train)
        pred = knn.predict(X_Test)

        print('WITH K=23')
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


        