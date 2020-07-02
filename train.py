import pandas as pd
import numpy as np
import pywt
from scipy import signal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.signal import argrelextrema
from pylab import *
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import pickle



Cgmvalues=[]
labels=[]

def myReadCSV(path):
    data = []

    with open(path, 'r') as f:
        for line in f.readlines():
            # line.strip()
            d = line.split(",")
            i, mean = 0, 0
            temp = []
            while (i < len(d)):
                if (d[i] == "NaN" or d[i] == "NaN\n" or d[i]=="NAN" or d[i]=="\n" or d[i]==""):
                    temp.append(mean)
                else:
                    temp.append(int(d[i]))
                    mean += (int(d[i]) - mean) / (i + 1)
                i += 1
            data.append(temp)
    return data

def getdata(Cgmvalues):
    for i in range(1,6):
        Cgmvalues.append(pd.read_csv(f"mealData{i}.csv", delimiter=',', dtype=np.float32))
    print("getdata fun running")






def getlabel(labels):
    mealData = []
    temp=[]
    mealAmount=[]
    for i in range(1, 6):
        #print(i)
        mealData.extend(myReadCSV("mealData" + str(i) + ".csv"))
        #print(len(mealData))
    for i in range(1,6):
        tempData=(myReadCSV("mealData" + str(i) + ".csv"))
        temp=myReadCSV("mealAmountData" + str(i) + ".csv")
        check = len(tempData)
        temp=temp[:check]
        for i in range(len(temp)):
            if(temp[i][0]==0):
                temp[i]=0
            elif( temp[i][0]>0 and temp[i][0]<=20):
                temp[i]=1
            elif( temp[i][0]>20 and temp[i][0]<=40):
                temp[i]=2
            elif( temp[i][0]>40 and temp[i][0]<=60):
                temp[i]=3
            elif( temp[i][0]>60 and temp[i][0]<=80):
                temp[i]=4
            elif( temp[i][0]>80 and temp[i][0]<=100):
                temp[i]=5
        #print(check)
        labels.extend(temp)
        



def pre_processing(Cgmvalues):
    for i in range(5):
        Cgmvalues[i] = Cgmvalues[i].interpolate(direction="both",method="linear",axis=0) 
    
    for i in range(5):
        Cgmvalues[i] = Cgmvalues[i].dropna()

    for i in range(5):
        Cgmvalues[i] = Cgmvalues[i].to_numpy()
    print("pre-processing fun running")


# 1st Feature
def fourier_transform(Cgmvalues):
    fourier_transform=list()
    for i in range(5):
        person_ft=list()
        for r in range(len(Cgmvalues[i])):
            temp=np.fft.irfft(Cgmvalues[i][r],n=2)
            #print(temp)
            person_ft.append(temp)
        fourier_transform.append(person_ft)
    print("fourierTransform-Completed")
    return np.array(fourier_transform)
    


# 2nd feature

def mealVelocity(Cgmvalues):
    meal_velocity=list()
    for i in range(5):
        personMeal=Cgmvalues[i]
        personMealVel=list()
        k=4
        for r in range(len(personMeal)):
            MealVelWin=list()
            for c in range(4,len(personMeal[r])):
                MealVelWin.append((personMeal[r][c]-personMeal[r][c-4])/4)
            personMealVel.append(MealVelWin)
        meal_velocity.append(personMealVel)
    print("MealVelocity-Completd")
    return(np.array(meal_velocity))


#3rd feature
def MOV(Cgmvalues):
    mov = []
    for i in range(5):
        personCgmvalues = Cgmvalues[i]
        person_MOV = []
        window = 4
        for row_i in range(len(personCgmvalues)):
            person_MOVpoint = []
            for col_i in range(window, len(personCgmvalues[row_i])):
                person_MOVpoint.append(
                  sum(personCgmvalues[row_i][col_i - window : col_i + 1]) / window)
            person_MOV.append(person_MOVpoint)
        mov.append(person_MOV)
    print("MOV runnning")
    return np.array(mov)



#4th feature
def rootMeanSquare(Cgmvalues):
    rmsvelocity=list()
    for i in range(5):
        personCgmvalues=Cgmvalues[i]
        rmsperson=list()

        for r in range(len(personCgmvalues)):
            rmsVelWin=list()
            for c in range(0,len(personCgmvalues[i]),4):
                if c+4 <len(personCgmvalues[i]):
                    temp=sum(personCgmvalues[r][c:c+4])/4
                    rmsVelWin.append(np.sqrt(temp))
                else:
                    temp=sum(personCgmvalues[r][c:])/(len(personCgmvalues[i])-c)
                    rmsVelWin.append(np.sqrt(temp))
            rmsperson.append(rmsVelWin)
        rmsvelocity.append(rmsperson)
    print("rms runniung")
    return np.array(rmsvelocity)


def featureMatrix(Mov,FFT,RMS,MEALVEL):
   
    feature_MOV=pd.DataFrame()
    for personLOC in Mov:
        feature_MOV = feature_MOV.append(personLOC, ignore_index=True)
        feature_MOV = feature_MOV.apply(abs)
    feature_FFT = pd.DataFrame()
    for personFFTS in FFT:
        feature_FFT = feature_FFT.append(personFFTS, ignore_index=True)
        feature_FFT = feature_FFT.apply(abs)
    featureRMS = pd.DataFrame()
    for personRMS in RMS:
        feature_RMS = featureRMS.append(personRMS, ignore_index=True)
        feature_RMS = feature_RMS.apply(abs)
    featureMealVel = pd.DataFrame()
    for personMealVel in MEALVEL:
        feature_MealVel = featureMealVel.append(personMealVel, ignore_index=True)
    featurematrix = pd.concat((feature_MOV, feature_FFT, feature_RMS,feature_MealVel), axis=1, ignore_index=True)
    featurematrix = featurematrix.dropna(axis=1)
    print("featureMatrix-Completed")
    return featurematrix

def cluster(inputdata):
    dbscan = DBSCAN(eps=150, min_samples=10).fit(inputdata)
    kmeans = KMeans(n_clusters=6, random_state=0).fit(inputdata)
    print("dbscan labels")
    print("kmeans-label")
    
    return kmeans.labels_,dbscan.labels_

def myLabelMapping(clusterLabel,actualLabel):
    print("myLabelMapping-started")
    print("clusterLabel")
    print("actualLabel")
    a=[]
    b=[]
    c=[]
    dBin=[]
    e=[]
    f=[]
    for i in range(len(clusterLabel)):
        if (clusterLabel[i]==0):
            a.append(i)
        elif(clusterLabel[i]==1):
            b.append(i)
        elif(clusterLabel[i]==2):
            c.append(i)
        elif(clusterLabel[i]==3):
            dBin.append(i)
        elif(clusterLabel[i]==4):
            e.append(i)
        elif(clusterLabel[i]==5):
            f.append(i)
    labels=[]
    d = defaultdict(int)
    if(len(a)!=0):
        for i in a:
            temp=actualLabel[i]
            d[temp]+=1
        result = max(d.items(), key=lambda x: x[1])
        labels.append(result[0])
    else:
        labels.append(0)
    d = defaultdict(int)
    if(len(b)!=0):
        for i in b:
            temp=actualLabel[i]
            d[temp]+=1
        result = max(d.items(), key=lambda x: x[1])
        labels.append(result[0])
    else:
        labels.append(0)
    d = defaultdict(int)
    if(len(c)!=0):
        for i in c:
            temp=actualLabel[i]
            d[temp]+=1
        result = max(d.items(), key=lambda x: x[1])
        labels.append(result[0])
    else:
        labels.append(0)
    d = defaultdict(int)
    if(len(dBin)!=0):
        for i in dBin:
            temp=actualLabel[i]
            d[temp]+=1
        result = max(d.items(), key=lambda x: x[1])
        labels.append(result[0])
    else:
        labels.append(0)
    d = defaultdict(int)
    if(len(e)!=0):
        for i in e:
            temp=actualLabel[i]
            d[temp]+=1
        result = max(d.items(), key=lambda x: x[1])
        labels.append(result[0])
    else:
        labels.append(0)
    d = defaultdict(int)
    if(len(f)!=0):
        for i in f:
            temp=actualLabel[i]
            d[temp]+=1
        result = max(d.items(), key=lambda x: x[1])
        labels.append(result[0])
    else:
        labels.append(0)
    finalLabel=[]
    for i in range(len(clusterLabel)):
        temp=clusterLabel[i]
        finalLabel.append(labels[temp])
    return finalLabel

def KNN(X_train, X_test, y_train, y_test,FILEPATH):
    classifier = KNeighborsClassifier(n_neighbors=25)
    classifier.fit(X_train, y_train)
    pickle.dump(classifier, open(FILEPATH, 'wb'))


def pca(featureMatrix,labels):
    data= np.array(featureMatrix)
    pca = PCA(n_components=5)
    pca.fit(data)
    updatedFeatureVector=pca.transform(data)
    pickle.dump(pca, open("xyz.pkl", 'wb'))
    pca_df = pd.DataFrame(pca.components_, index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5'])
    plt.bar(['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5'], pca.explained_variance_ratio_)
    splitSize = int(len(updatedFeatureVector) * 1)
    trainData = updatedFeatureVector[:splitSize]
    trainLable = labels[:splitSize]
    testData = updatedFeatureVector[splitSize:]
    testLable = labels[splitSize:]
    print("train label")
    print("testlabel")
    trainLabelKmeans,trainLabelDBSCAN=cluster(trainData)
    finalLabelK_Means=myLabelMapping(trainLabelKmeans,trainLable)
    finalLabelDBSCAN=myLabelMapping(trainLabelDBSCAN,trainLable)  
    Knn2=KNN(trainData,testData,finalLabelDBSCAN,testLable,"DBSCAN2.pkl")
    Knn1=KNN(trainData,testData,finalLabelK_Means,testLable,"K_MEANS2.pkl")
    print("TRAINING COMPLETE")
    

getdata(Cgmvalues)
getlabel(labels)

pre_processing(Cgmvalues)


FFT=fourier_transform(Cgmvalues)
MEALVEL= mealVelocity(Cgmvalues)
RMS= rootMeanSquare(Cgmvalues)
Mov=MOV(Cgmvalues)


FeatureMatrix=featureMatrix(Mov,FFT,RMS,MEALVEL)
pca(FeatureMatrix,labels)




