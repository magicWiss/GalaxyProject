'''
Classe per la principal component analysis.
Riduzione delle dimensioni da 160*160 in 125 features.


'''
from sklearn.decomposition import PCA

class PrincipalComonentAnalysis:
    def __init__(self,components):
        self.components=components
        self.pca=PCA()

    def reduceComponents(self,dataSet):
        return self.pca.fit_transform(dataSet)
