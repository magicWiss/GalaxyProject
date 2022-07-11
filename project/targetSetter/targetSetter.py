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
with open ("targetSetter/training_labels.csv","w", newline="") as f:
    writer=csv.writer(f)
    print("Scrittura headers")
    writer.writerow(("imageId","label"))
    print("Scrittura elementi")
    for i in imageTolabel:
        print("Scrittura elemento:",i)
        writer.writerow(i)



