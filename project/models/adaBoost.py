# tune regularization for multinomial logistic regression
import numpy as np
from numpy import mean
from numpy import std

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot

class AdaBoost:

    def __init__(self):
        pass

    def get_models(self):
            
            models = dict()
            # define number of trees to consider
            n_trees = [10, 50, 100, 500, 1000, 5000]
            for n in n_trees:
                models[str(n)] = AdaBoostClassifier(n_estimators=n)
            return models   
        
        # evaluate a give model using cross-validation
    def evaluate_model(self,model, X_Train, Y_Train,X_Test,Y_Test):
            # define the evaluation procedure
            #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            # evaluate the model
            #scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
            model.fit(X_Train,np.ravel(Y_Train))
            y_pred = model.predict(X_Test)  
            score=accuracy_score(y_pred,Y_Test)
            print("SCORE= ",score)
            return score
    def predict(self, X_Train,Y_Train, X_Test, Y_Test):
        models=self.get_models()
 

 
        # define dataset
       
        # get the models to evaluate
        
        # evaluate the models and store results
        results, names = list(), list()
        for name, model in models.items():
            # evaluate the model and collect the scores
            scores = self.evaluate_model(model, X_Train, Y_Train,X_Test, Y_Test)
            # store the results
            results.append(scores)
            names.append(name)
            # summarize progress along the way
            print('>%s %.3f ' % (name, mean(scores)))
        # plot model performance for comparison
        