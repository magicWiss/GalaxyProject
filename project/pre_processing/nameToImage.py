class NameImage:

    def __init__(self) -> None:
        
        self.basicPath='images/training/'
        self.extension='.jpg'


    def nameToImage(self,name):
        return self.basicPath+name+self.extension