'''
classe di routing del modello da utilizzare nel run corrente.
In base al valore definito nel file di parametri definisce quale modello istanziare
La classe è sicuramente da migliorare in quanto sicuramente dovranno essere passati al metodo trainModel anche i vari
Y_training, X_training etc...
'''

from models.k_means import Kmeans
from models.knn import KNN
class ModelRouter:

    def __init__(self, type):

        self.type=type          #il valore di type definisce il modello da invocare

    def trainModel(self, X_Training, Y_Training, X_Test, Y_Test):

        if self.type=='kmeans':
            model=Kmeans()
            model.predict(X_Training)
        
        elif self.type=='knn':
            model=KNN()
            model.predict(X_Training,Y_Training,X_Test,Y_Test)
