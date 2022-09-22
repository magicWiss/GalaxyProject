# tune regularization for multinomial logistic regression
import numpy as np
from numpy import mean
from numpy import std

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot

class LogisticReg:

    def __init__(self):

        pass
                # get a list of models to evaluate
    def get_models(self):
            models = dict()
            for p in [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]:
                # create name for model
                key = '%.4f' % p
                # turn off penalty in some cases
                if p == 0.0:
                    # no penalty in this case
                    models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none',max_iter = 100000000)
                else:
                    models[key] = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p,max_iter = 100000000)
            return models    
        
        # evaluate a give model using cross-validation
    def evaluate_model(self,model, X_Train, Y_Train,X_Test,Y_Test):
            # define the evaluation procedure
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
        