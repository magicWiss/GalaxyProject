
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
class RandomForest:

    def __init__(self):

        pass

    def predict(self, X_Train,Y_Train, X_Test, Y_Test):
        print("\nAddestramento di un modello KNN")
        #qui si implementa il metodo per l'addesteamento del modello
        #si inseriscono qui le operazioni di:
        #PCA e scelta del numero di componenti ottimali
        #cross-validation per la scelta del numero di k ottimi per il modello
        #stampa del risultato dell'addestramento
        model=KNeighborsClassifier()
        model.fit(X_Train,Y_Train)
        predicted_test=model.predict(X_Test)
        acc=accuracy_score(Y_Test,predicted_test)

        print("KNN Accuricy:",acc)
        print(confusion_matrix(Y_Test, predicted_test))
        print(classification_report(Y_Test, predicted_test))


        