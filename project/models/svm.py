from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import pandas as pd


class SVM:

    def __init__(self):

        pass

    def predict(self, X_Train,Y_Train, X_Test, Y_Test):

     print("\nAddestramento di un modello SVM")
    
    #Per utilizzare il kernel gaussiano, è necessario specificare 'rbf' come valore per il parametro Kernel della classe SVC.
     svclassifier = SVC(kernel='rbf')
     svclassifier.fit(X_Train, Y_Train)

     #previsione e valutazione
     y_pred = svclassifier.predict(X_Test)
     print(confusion_matrix(Y_Test, y_pred))
     print(classification_report(Y_Test, y_pred))
     print("accuracy score:\n")
     