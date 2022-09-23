from turtle import color
from matplotlib import  cm, pyplot as plt
import pandas as pd
import numpy as np
import math
#i risultati sono in una lista
#svm->[bestResPP1,bestResPP2,bestResPP3]

if __name__=='__main__':

    svmRes=[0.49, 0.59, 0.45]
    adaBoost=[0.52,0.57,0.45]
    logReg=[0.53,0.62,0.50]
    randomForest=[0.57,0.62,0.52]
    gaussianCluster=[0.32,0.15,0.07]
    knn=[0.40,0.32,0.30]
    kmeans=[0.31,0.22,0.2]

    models=['svm','adaBoost','logReg','randomForest','gaussianCluster','knn','kmeans']
    pp1=[0.49,0.52,0.53,0.57,0.32,0.40,0.31]
    pp2=[0.59,0.57,0.62,0.62,0.15,0.32,0.22]
    pp3=[0.45,0.45,0.50,0.52,0.07,0.30,0.20]
    dataMap={"models":models,"pp1":pp1,"pp2":pp2,"pp3":pp3}
    data=pd.DataFrame(data=dataMap,columns=['models','pp1','pp2','pp3'])
    print (data)


    #equazione per il calcolo del più performate
    meanpp1=np.mean(data['pp1'])
    meanpp2=np.mean(data['pp2'])
    meanpp3=np.mean(data['pp3'])

    print("Media pp1", meanpp1)
    
    print("Media pp2", meanpp2)
    print("Media pp3", meanpp3)

    colorScoreMean=(1*np.mean(data['pp2'])+0.5*np.mean(data['pp1']))/1.5
    formScoreMean=(0.5*np.mean(data['pp2'])+1*np.mean(data['pp1'])+1*np.mean(data['pp3']))/2.5
    print("COLOR:",colorScoreMean)
    colorScore=1 / (1 + math.exp(-colorScoreMean))


    

    print(colorScore)
    print("FORM:",formScoreMean)
    formScoreMean=1/(1 + math.exp(-formScoreMean))
    print(formScoreMean)

    

    pp1BestNumber=len(data[data.pp1>0.5])
    pp2BestNumber=len(data[data.pp2>0.5])
    pp3BestNumber=len(data[data.pp3>0.5])
    
    pp1BestMean=np.mean(data['pp1'])
    pp2BestMean=np.mean(data['pp2'])
    pp3BestMean=np.mean(data['pp3'])

   
    
    print("numero models pp2 score >0.5  ",pp2BestNumber, " avg  ",pp2BestMean)
    
    print("numero models pp3 score >0.5  ",pp3BestNumber, " avg  ",pp3BestMean)

    fig, ax = plt.subplots()

    pipeline = ['Binario', 'Otsu', 'Canny']
    counts = [pp1BestNumber, pp2BestNumber, pp3BestNumber]
    bar_labels = ['Binario', 'Otsu', 'Canny']
    bar_colors = ['tab:red', 'tab:blue', 'tab:green']

    ax.bar(pipeline, counts, label=bar_labels, color=bar_colors)

    ax.set_ylabel('modelli con accuray > 0.55')
    ax.set_title('Performance modelli nelle pipeline')
    

    plt.show()

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    threshold = 0.55
    values = [pp1BestMean,pp2BestMean,pp3BestMean]
    values=np.array(values)
    x = ['Binario','Otsu','Canny']

    a_threshold = np.maximum(values - threshold, 0)
    b_threshold = np.minimum(values, threshold)

    fig, ax = plt.subplots()
    ax.bar(x, b_threshold, 0.35, color="blue")
    ax.bar(x, a_threshold, 0.35, color="yellow", bottom=b_threshold)

    plt.axhline(threshold, color='red', ls='dotted')

    plt.show()

    
