from decimal import Clamped

from numpy import maximum
from classes import Classes


class Node:
    
    def __init__(self, value, child = None, father=None) -> None:
      self.label = value
      self.children = set()
      self.father=father
      if child != None:
         for value in child:
            self.children.append(value)
    
    def addChilds(self,childs):
        self.children=childs

    def getChild(self,id):
        for i in self.children:
            if i.label==id:
                return i
    
    def addFather(self, father):
        self.father=father


class Tree:

    def __init__(self):
        self.root=Node(1)
        self.node0=Node(0)
        self.node2=Node(2)
        self.node3=Node(3)
        self.node4=Node(4)
        self.node4=Node(4)
        self.node53=Node(53)
        self.node54=Node(54)
        self.node61=Node(61)
        self.node62=Node(62)
        self.node63=Node(63)
        self.node64=Node(64)
        self.node7=Node(7)
        self.node8=Node(8)
        self.node9=Node(9)
        self.node10=Node(10)
        self.node11=Node(11)


        #adding relationships
        self.root.addChilds([self.node7,self.node2])

        #etichetta 1
        self.node7.addChilds([self.node61])

        self.node61.addChilds([self.node8])

        #etichetta 2

        self.node2.addChilds([self.node9, self.node3])

        self.node9.addChilds([self.node62])

        self.node62.addChilds([self.node8])

        #etichetta 3
        self.node3.addChilds([self.node4])

        self.node4.addChilds([self.node10,self.node54])

        self.node10.addChilds([self.node11])

        self.node11.addChilds([self.node53])

        self.node53.addChilds([self.node63])
        
        self.node63.addChilds([self.node8])

        
        
        self.node54.addChilds([self.node64])

        self.node64.addChilds([self.node8])


    


    def computeAnswer(self, answer):
        maximum=max(answer)
        index=answer.index(maximum)
        return index

#restituisce il cammino dal root al nodo foglia 

    def computePath(self, cammino, answers, node):
    
        if (node==None):
            return cammino
        
        if (node.label==1):
            maxIndex=self.computeAnswer(answers[:3])    #prima risposta
            if maxIndex==0:
                node=node.getChild(7)
                cammino.append(node.label)
            elif maxIndex==1:
                node=node.getChild(2)
                cammino.append(node.label)
            elif maxIndex==2:
                node=None

            return self.computePath(cammino, answers, node)
        

    #Etichetta 1
        if (node.label==7):
            maxIndex=self.computeAnswer(answers[15:18])    #prima risposta
            if maxIndex==0 or maxIndex==1 or maxIndex==2:
                node=node.getChild(61)
                cammino.append(node.label)
            return self.computePath(cammino, answers, node)

        if (node.label==61):
            maxIndex=self.computeAnswer(answers[13:15])    #prima risposta
            if maxIndex==0:
                node=node.getChild(8)
                cammino.append(node.label)
                return self.computePath(cammino, answers, node)

            elif maxIndex==1:
                return cammino

        if (node.label==8):
            node=None
            return cammino

    #Etichetta 2
        if (node.label==2):
            maxIndex=self.computeAnswer(answers[3:5])
            if maxIndex==0:
                node=node.getChild(9)
                cammino.append(node.label)
                return self.computePath(cammino, answers, node)
            if maxIndex==1:
                node=node.getChild(3)
                cammino.append(node.label)
                return self.computePath(cammino, answers, node)
        

        if (node.label==9):
            maxIndex=self.computeAnswer(answers[25:27])
            if maxIndex==0 or maxIndex==1:
                node=node.getChild(62)
                cammino.append(node.label)
                return self.computePath(cammino, answers, node)
        

        if (node.label==62):
            maxIndex=self.computeAnswer(answers[13:15])    #prima risposta
            if maxIndex==0:
                node=node.getChild(8)
                cammino.append(node.label)
                return self.computePath(cammino, answers, node)

            elif maxIndex==1:
                return cammino
        
        #etichetta 3
        if (node.label==3):         
            
            node=node.getChild(4)
            cammino.append(node.label)
            return self.computePath(cammino, answers, node)
        
        if (node.label==4):
            maxIndex=self.computeAnswer(answers[7:9])    #prima risposta
            if maxIndex==0:
                node=node.getChild(10)
                cammino.append(node.label)
                return self.computePath(cammino, answers, node)
            elif maxIndex==1:
                node=node.getChild(54)
                cammino.append(node.label)
                return self.computePath(cammino, answers, node)

        
        if (node.label==10):
           
            node=node.getChild(11)
            cammino.append(node.label)
            return self.computePath(cammino, answers, node)
        
        if (node.label==11):
            node=node.getChild(53)
            cammino.append(node.label)
            return self.computePath(cammino, answers, node)
        
        if (node.label==53):
            node=node.getChild(63)
            cammino.append(node.label)
            return self.computePath(cammino, answers, node)
        
        if (node.label==63):
            node=node.getChild(8)
            cammino.append(node.label)
            return self.computePath(cammino, answers, node)

        #etichetta 4

        
            
        
        if (node.label==54):
            node=node.getChild(64)
            cammino.append(node.label)
            return self.computePath(cammino, answers, node)
        
        if (node.label==64):
            node=node.getChild(8)
            cammino.append(node.label)
            return self.computePath(cammino, answers, node)








    def computeTargetValue(self, id, answers):

        currentClass=Classes(id)                        #creo l'oggetto class
        path=[self.root.label]                          #il cammino iniziale ha solo il root
        self.computePath(path,answers, self.root)       #calcolo il cammino
        print(path)

        currentClass.cammino=path
        currentClass.setTargetValue()                   #calcolo il target associato
        return currentClass                             #resituisco la classe corrente          
 





