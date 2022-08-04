#classe specializzata nella lettura e scrittura dei file csv
import csv

from classes.pattern import Pattern
import sys

class CSVwriter:

    def __init__(self):
        self.patternsPreOne='project/files/patternsOne.csv'            #file dei pattern con il preprocessamento 1
        self.patternsPreTwo='project/files/patternsTwo.csv'            #file dei pattern con il preprocessamento 2
        self.patternsOnePCA='project/files/patternsOnePCA.csv'         #file dei pattern con pre 1 e dopo PCA
        self.patternTwoPCA='project/files/patternsTwoPCA.csv'          #file dei pattern con pre 2 e dopo PCA

    def printLoadingBar(count,total,suffix):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('PREPROCESSING IMAGE:IN PROGRESS: [%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()  

    def definePath(self,type,pcaApplied):
        path=''
        #scrivo su patternsOne.csv
        if type==1 and pcaApplied==False:
            path=self.patternsPreOne
        
        elif type==2 and pcaApplied==False:
            path=self.patternsPreTwo

        elif type==1 and pcaApplied==True:
            path=self.patternsOnePCA

        elif type==2 and pcaApplied==True:

            path=self.patternTwoPCA

        return path

    def read(self, path):
        output=[]
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                #pattern=Pattern(id=row[0],fullPath=row[1],features=row[2],labe=row[3])
                print(row)


          
                
    def write(self, path, data):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            total=len(data)
            count=0

            for i in data:
                self.printLoadingBar(count,suffix,total)
                info=i.getDataForCsv()
                writer.writerow(info)
                count+=1
    


    def writePatterns(self,type,pcaApplied,data):
        path=self.definePath(type,pcaApplied)

        

        #scrittura
        self.write(path,data)  

    def readPatterns(self,type,pcaApplied):
        path=self.definePath(type,pcaApplied)
        return self.read(path)

    

      


            

