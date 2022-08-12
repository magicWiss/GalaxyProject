from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from matplotlib import  cm, pyplot as plt
import seaborn as sns
class Kmeans:

    def __init__(self):

        pass

    def getBestScore(self, scores):
        print("Miglior score del k-means")
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
        plt.axvline(1, color = "r")
        plt.show()
        
    

    def predict(self, data):
        print("Addestramento di un modello kmeans")
        #siluette score per ogni k
        silhoette_score={}              #array di 4 posizione index->clusters(0->2, 1->3, 2->4, 3->5)
        for k in range(2,10):
            model_kmeans_k = KMeans( n_clusters = k)
            model_kmeans_k.fit(data)
            labels_k = model_kmeans_k.labels_
            score_k = silhouette_score(data,labels_k)
            silhoette_score[k] = score_k
            print("Tested kMeans with k = %d\tSS: %5.4f" % (k, score_k))


        best=self.getBestScore(silhoette_score)
        self.printShilutte(silhoette_score,best)
        
       
        