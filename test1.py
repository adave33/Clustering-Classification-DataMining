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
    for i in range(1):
        Cgmvalues.append(pd.read_csv(f"mealData1.csv", delimiter=',', dtype=np.float32))
    print("getdata fun running")
    

def pre_processing(Cgmvalues):
    Cgmvalues[0] = Cgmvalues[0].interpolate(direction="both",method="linear",axis=0) 
    
    
    Cgmvalues[0] = Cgmvalues[0].dropna()

    
    Cgmvalues[0]= Cgmvalues[0].to_numpy()
    print("pre-processing fun running")



# 1st Feature
def fourier_transform(Cgmvalues):
    temp=list()
    
    person_ft=list()
    for r in range(len(Cgmvalues[0])):
        temp=np.fft.irfft(Cgmvalues[0][r],n=2)
        #print(temp)
        person_ft.append(temp)
    print("fourierTransform-Completed")
    return np.array(person_ft)

# 2nd feature

def mealVelocity(Cgmvalues):
    meal_velocity=list()
    personMeal=Cgmvalues[0]
    personMealVel=list()
    for r in range(len(personMeal)):
        MealVelWin=list()
        for c in range(4,len(personMeal[r])):
            MealVelWin.append((personMeal[r][c]-personMeal[r][c-4])/4)
    meal_velocity.append(MealVelWin)
    print("MealVelocity-Completd")
    return(np.array(meal_velocity))



#3rd feature
def MOV(Cgmvalues):
    mov = []
    personCgmvalues = Cgmvalues[0]
    person_MOV = []
    window = 4
    for row_i in range(len(personCgmvalues)):
        person_MOVpoint = []
        for col_i in range(window, len(personCgmvalues[row_i])):
          person_MOVpoint.append(
              sum(personCgmvalues[row_i][col_i - window : col_i + 1]) / window)
        person_MOV.append(person_MOVpoint)
    print("mov running")
    return person_MOV
   

#4th feature


def rootMeanSquare(Cgmvalues):
    rmsvelocity=list()
    personCgmvalues=Cgmvalues[0]
    rmsperson=list()

    for r in range(len(personCgmvalues)):
        rmsVelWin=list()
        for c in range(0,len(personCgmvalues[0]),4):
            if c+4 <len(personCgmvalues[0]):
                temp=sum(personCgmvalues[r][c:c+4])/4
                rmsVelWin.append(np.sqrt(temp))
            else:
                temp=sum(personCgmvalues[r][c:])/(len(personCgmvalues[0])-c)
                rmsVelWin.append(np.sqrt(temp))
    rmsvelocity.append(rmsVelWin)
    print("rms runniung")
    print(rmsvelocity)
    return rmsvelocity
    



def featureMatrix(Mov,FFT,RMS,MEALVEL):
   
    feature_MOV=pd.DataFrame(np.array(Mov))
    feature_FFT = pd.DataFrame(np.array(FFT))
    feature_RMS = pd.DataFrame(np.array(RMS))
    featureMealVel = pd.DataFrame(np.array(MEALVEL))
    featurematrix = pd.concat((feature_MOV, feature_FFT, feature_RMS,featureMealVel), axis=1, ignore_index=True)
    print(featurematrix.shape)
    featurematrix = featurematrix.dropna(axis=1)
    print(featurematrix.shape)
    print("featureMatrix-Completed")
    print(featurematrix)
    return featurematrix

    
getdata(Cgmvalues)
pre_processing(Cgmvalues)
FFT=fourier_transform(Cgmvalues)
MEALVEL= mealVelocity(Cgmvalues)
RMS= rootMeanSquare(Cgmvalues)
Mov=MOV(Cgmvalues)
FeatureMatrix=featureMatrix(Mov,FFT,RMS,MEALVEL)


model1=pickle.load(open('xyz.pkl', 'rb'))
updatedFeatures=model1.transform(FeatureMatrix)
model2=pickle.load(open('K_MEANS2.pkl', 'rb'))
model3=pickle.load(open('DBSCAN2.pkl', 'rb'))

predictedLabels1=model2.predict(updatedFeatures)
predictedLabels2=model3.predict(updatedFeatures)


outputMatrix=[]

for i in range(len(predictedLabels2)):
    temp=[]
    temp.append(predictedLabels1[i])
    temp.append(predictedLabels2[i])
    outputMatrix.append(temp)

print("Column 0:K_Means, Column 1:DBSCAN")
print(outputMatrix)





