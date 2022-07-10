class Classes:
    
   
    def __init__(self, id) -> None:
        
        self.id=id
        self.cammino=[]
        self.target=""


        self.class0=[1]              #min nodes that are unique for the class
        self.class1=[1,7]            #real path is (1,2,6,[8])
        
        self.class2=[1,2,9]         #real path is (1,2,9,6,[8])
        
        self.class3=[1,2,3,43,10]    #reqal path is (1,2,3,4,10,11,5,6,[8])

        self.class4=[1,2,3,44,5]     #real path is (1,2,3, 4, 5,6,[8])

    
    #verifico che il cammino da root a foglia del nodo corrente sia intertno alla classe corrente    
    def compareList(self, path, classe):
        s1=" ".join(str(i) for i in classe)
        s2=" ".join(str(i) for i in path)
        if s1 in s2:
            return True
        else:
            return False

    def setTargetValue(self):
        
        if self.compareList(self.cammino,self.class0):
            self.target="0"
        
        if self.compareList(self.cammino,self.class1):
            self.target="1"
        
        if  self.compareList(self.cammino,self.class2):
            self.target="2"
        if self.compareList(self.cammino,self.class3):
            self.target="3"
        if self.compareList(self.cammino,self.class4):
            self.target="4"