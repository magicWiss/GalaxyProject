from matplotlib import  cm, pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

class ViewData:


    #method->3 (3d), 2 (2d)
    def __init__(self, method):

        self.method=method

    def visualize(self,data,labels):
        print("\n\n=================================================")
        print("===============PLOT TRAINING DATA================")
        print("=================================================")
        
        if self.method==3:
            self.visualize3D(data,labels)
        else:
            self.visualize2D(data,labels)

    def visualize2D(self,featuresNorm,labels):
        pca=PCA(n_components=2)
        principalComponents=pca.fit_transform(featuresNorm)

        #dataframe conententi le due componenti estratte
        principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])
        #final df con cui effettuare la visualizzazione

        ex_variance=np.var(principalComponents,axis=0)
        ex_variance_ratio = ex_variance/np.sum(ex_variance)
        print("Variance:",ex_variance_ratio)
        
        finalDf = pd.concat([principalDf, labels], axis = 1)
        
        
        fig = plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
    
        ax.set_title('3 component PCA', fontsize = 20)
        targets = [0, 1, 2,3,4]
        colors = ['r', 'g', 'b','y']
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                
                    , c = color
                    , s = 50)
        ax.legend(targets)
        ax.grid()
        plt.show()

    def visualize3D(self,featuresNorm,labels):
        pca=PCA(n_components=3)
        principalComponents=pca.fit_transform(featuresNorm)
        ex_variance=np.var(principalComponents,axis=0)
        ex_variance_ratio = ex_variance/np.sum(ex_variance)
        print("Variance:",ex_variance_ratio)

        #dataframe conententi le due componenti estratte
        principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2','principal component 3'])
        finalDf = pd.concat([principalDf, labels], axis = 1)

        targets = [0, 1, 2,3,4]
        colors = ['r', 'g', 'b','y']
        

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection='3d')

        fig.patch.set_facecolor('white')
        for target, color in zip(targets,colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , finalDf.loc[indicesToKeep, 'principal component 3']          
                    , c = color
                    , s = 50)

            
        # for loop ends
        ax.set_xlabel("First Principal Component", fontsize=14)
        ax.set_ylabel("Second Principal Component", fontsize=14)
        ax.set_zlabel("Third Principal Component", fontsize=14)

        ax.legend()
        plt.show()
