from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from matplotlib import  cm, pyplot as plt
import seaborn as sns
import numpy as np
class GaussMix:

    def __init__(self):

        pass
    
    def getBestScore(self, scores):
        print("Miglior score del Gaussian Mixture")
        minval = max(scores.values())
        res = [k for k, v in scores.items() if v==minval]
        print("Score:",minval)
        print("K used:",res)
        return res[0]
    
    def printShilutte(self, scores,best):
      
        plt.figure(figsize = (16,5))
        plt.plot(scores.values())
        plt.xticks(range(0,len(scores.keys()),1), scores.keys())
        plt.title("Silhouette Metric")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.axvline(best, color = "r")
        plt.show()

    def predict(self, data):
        print("Addestramento di un modello Gaussian Mixture")
        #siluette score per ogni k
        # define the model
        score={} 

        for k in range (2,10):
            gaussian_model = GaussianMixture(n_components=k)

            # train the model
            gaussian_model.fit(data)

            # assign each data point to a cluster
            gaussian_result = gaussian_model.predict(data)

        
          
            
            score_k = silhouette_score(data,gaussian_result)
            
            print("Tested kMeans with k = %d\tSS: %5.4f" % (k, score_k))
            score[k]=score_k


        best=self.getBestScore(score)
        self.printShilutte(score,best)
       
        
       
        