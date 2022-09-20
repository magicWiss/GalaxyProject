from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from matplotlib import  cm, pyplot as plt
import seaborn as sns
class Kmeans:

    def __init__(self):

        pass

    def getBestScore(self, scores):
        print("Miglior score del k-means")
        bestK=[0,0]
        best1,best2=0,0
        for k in scores:
            current=scores[k]
            if current[1]>=best1:
                bestK.insert(0,current)
                best2=best1
                best1=current[1]

            elif current[1]<best1 and current[1]>=best2:
                bestK.insert(1,current)
                best2=current[1]

            bestK=bestK[:2]



    def printShilutte(self, scores,best):
      
        plt.figure(figsize = (16,5))
        plt.plot(scores.values())
        plt.xticks(range(0,len(scores.keys()),1), scores.keys())
        plt.title("Silhouette Metric")
        plt.xlabel("k")
        plt.ylabel("Silhouette")
        plt.axvline(1, color = "r")
        plt.show()
        
    
    '''def printCluster(self, best):
        for i in best:
            model=i[0]
            label=model.labels_
            #Getting the Centroids
            centroids = model.cluster_centers_
            import numpy as np
            import pandas as pd
            u_labels = np.unique(label)
            
            #plotting the results:
            
            for i in u_labels:
                plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
            plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k)
            plt.legend()
            plt.show()'''

    def predict(self, data):
        print("Addestramento di un modello kmeans")
        #siluette score per ogni k
        silhoette_score={}              
        for k in range(2,6):
            model_kmeans_k = KMeans( n_clusters = k, init='k-means++', n_init=10, max_iter=6000, tol=0.0001, verbose=0, random_state=None, copy_x=True)
            model_kmeans_k.fit(data)
            labels_k = model_kmeans_k.labels_
            score_k = silhouette_score(data,labels_k)
            silhoette_score[k] = (model_kmeans_k,score_k)
            
            if k==2:
                self.printShilutte(silhoette_score,2)

            #print("Tested kMeans with k = %d\tSS: %5.4f" % (k, score_k))


        #best=self.getBestScore(silhoette_score)
        
        
        
                
                
                
            
                