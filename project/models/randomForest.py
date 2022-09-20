
from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

class RandomForest:

    def __init__(self):

        pass

    def predict(self, X_Train,Y_Train, X_Test, Y_Test):
        print("\nAddestramento di un modello RANDOM FOREST")
        #combinare più alberi decisionali nel determinare l'output finale piuttosto che fare affidamento su singoli alberi decisionali.
        #scgliamo 100 alberi
        model = RandomForestClassifier(n_estimators = 200, criterion='entropy',random_state = 0)

        #fit
      
        model.fit(X_Train,np.ravel(Y_Train) )  

        #Previsione di un nuovo risultato cambiando i valori
        y_pred = model.predict(X_Test)  


        cm=confusion_matrix(Y_Test,y_pred)
        print(cm)

        print("Accuracy:",accuracy_score(Y_Test, y_pred))
        model.score(X_Test, Y_Test)


    

        