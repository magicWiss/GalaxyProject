from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import  cm, pyplot as plt
import seaborn as sns
class Kmeans:

    def __init__(self):

        pass

    def predict(self, data):
        print("Addestramento di un modello kmeans")
        #siluette score per ogni k
        max_k=7
        silhoette_scores=[0,0]             
        for k in range(2,max_k):
            model_kmeans_k = KMeans( n_clusters = k, init='k-means++', n_init=10, max_iter=60000, tol=0.0001, verbose=0, random_state=None, copy_x=True)
            model_kmeans_k.fit(data)
            labels_k = model_kmeans_k.labels_
            score_k = silhouette_score(data,labels_k)
            silhoette_scores.insert(k,score_k)


        print("\nTracciamo un grafico a linee del tasso di errore")
        plt.figure(figsize=(10,6))
        plt.plot(range(2,max_k,1),silhoette_scores[2:],color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Shiluette score vs. K centroidi')
        plt.xlabel('K')
        plt.ylabel('Shiluette score')
        plt.show()

        #miglior valore
        best_k=silhoette_scores.index(max(silhoette_scores))
        model_kmeans_k = KMeans( n_clusters = best_k, init='k-means++', n_init=10, max_iter=60000, tol=0.0001, verbose=0, random_state=None, copy_x=True)
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        labelsComp=pd.DataFrame(model_kmeans_k.fit_predict(data),columns=['target'])
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data)
        
        from sklearn.preprocessing import StandardScaler
        scalar=StandardScaler()
        finalDf = pd.DataFrame(data = scalar.fit_transform(pca_result)
                , columns = ['pc1', 'pc2','pc3'])
        finalDf=pd.concat([finalDf, labelsComp], axis = 1)
       
        

        
        colors = ['r', 'g']
        targets = set(model_kmeans_k.labels_)
        
            
        if best_k==3:
            colors.append('b')
        elif best_k==4:
            colors.append('y')
            colors.append('b')
        elif best_k==5:
            colors.append('y')
            colors.append('b')
            colors.append('g')
        elif best_k==6:
            colors.append('y')
            colors.append('b')
            colors.append('g')
            colors.append('p')


        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection='3d')

        fig.patch.set_facecolor('white')
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
                    , finalDf.loc[indicesToKeep, 'pc2']
                    , finalDf.loc[indicesToKeep, 'pc3']          
                    , c = color
                    , s = 50)

            
        # for loop ends
        ax.set_xlabel("First Principal Component", fontsize=14)
        ax.set_ylabel("Second Principal Component", fontsize=14)
        ax.set_zlabel("Third Principal Component", fontsize=14)

        ax.legend(targets)
        plt.show()
        

        print("Score",max(silhoette_scores))
        
        '''cen_x = [i[0] for i in centroids] 
        cen_y = [i[1] for i in centroids]
        ## add to df
        data['cen_x'] = data.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
        data['cen_y'] = data.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
        # define and map colors
        colors = ['#DF2020', '#81DF20', '#2095DF']
        data['c'] = data.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})'''



            
            



        
        
        
                
                
                
            
                