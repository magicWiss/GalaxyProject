#LETTURA DEL FILE CSV "project\targetSetter\training_solutions_rev1.csv" e creazione di un csv con IDimage-label

import csv
from decisionTree import Tree
file= open("targetSetter/training_solutions_rev1.csv")

#lettura del file csv

csvreader=csv.reader(file)

#estrazione dell'header
header=[]
header=next(csvreader)

#creazione dell'istanza del decision tree da utilizzare
decisionTree=Tree()

#lista che includerà tutti le coppie img,etichetta
imageTolabel=[]

for row in csvreader:

    
    
    id=row[0]
    answer=row[1:]
    currentClass=decisionTree.computeTargetValue(id,answer)
    imageLabel=(currentClass.id,currentClass.target)
    imageTolabel.append(imageLabel)
print("Sono stati identificati",str(len(imageTolabel))," records")
for i in range(0,100):
    print("Elemento ", i, "-> ", imageTolabel[i])


file.close()

popClass1=0
popClass2=0
popClass3=0
popClass4=0
popClass0=0
with open ("targetSetter/training_labels.csv","w", newline="") as f:
    writer=csv.writer(f)
    print("Scrittura headers")
    writer.writerow(("imageId","label"))
    print("Scrittura elementi")
    for i in imageTolabel:
        if i[1]=="0":
            popClass0+=1
        
        elif i[1]=="1":
            popClass1+=1
            
        elif i[1]=="2":
            popClass2+=1
            
        elif i[1]=="3":
            popClass3+=1
            
        elif i[1]=="4":
            popClass4+=1
        print("Scrittura elemento:",i)
        writer.writerow(i)

with open ("targetSetter/summary.csv","w", newline="") as f:

    writer=csv.writer(f)
    print("Scrittura headers")
    writer.writerow(("class","popolation","totale"))
    writer.writerow(("Class 0",popClass0,""))
    writer.writerow(("Class 1",popClass1,""))
    writer.writerow(("Class 2",popClass2,""))
    writer.writerow(("Class 3",popClass3,""))
    writer.writerow(("Class 4",popClass4,""))
    writer.writerow(("Totale","",popClass1+popClass2+popClass3+popClass4+popClass0))
    


