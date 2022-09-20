
from sklearn.model_selection import KFold, cross_val_score

#I dati di training utilizzati nel modello vengono suddivisi, in k numero di insiemi più piccoli, da utilizzare per convalidare il modello. 
# Il modello viene quindi addestrato su k-1 sottoinsiemi del set di allenamento. Il sottoinsieme rimanente viene quindi utilizzato come set 
# di convalida per valutare il modello.

class CrossValidation:

    def __init__(self, n_split) :
        self.k_folds = KFold(n_splits = n_split)

    
    def kFoldCV(self, X_train, Y_train, modello):
        #Ora valutiamo il nostro modello e vediamo come si comporta su ogni k-fold.

        scores = cross_val_score(modello, X_train, Y_train)
        return scores
        #print("Cross Validation Scores: ", scores)
        #print("Average CV Score: ", scores.mean())
        #print("Number of CV Scores used in Average: ", len(scores))  
        