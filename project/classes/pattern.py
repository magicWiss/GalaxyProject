#Classe che rappresenta un img in features.
#E' caratterizzata da:
#                   1)nome dell'immagine
#                   2)lista di features
#                   3)nome completo (path) dell'immagine
#                   4)label associata


import sys
import numpy as np

class Pattern:

    def __init__(self, id,fullPath, features,label):
        self.id=id
        self.fullPath=fullPath
        self.features=features
        self.label=label

    def getDataForCsv(self):
        
        features=np.array(self.features)
        np.set_printoptions(threshold=sys.maxsize)
        return [self.id,self.fullPath,features,self.label]


    
   


