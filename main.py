#Model is generally referred to the Database, if any. Here, we are not using any Databases. 
#View is referred to the API, which accepts a request and returns a response. Controller is 
#the part that does all the heavy work, i.e, data processing, building classifier, etc. Hence, 
#it is known as the MVC Architecture!

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

#This line imports the library which stores the letters.
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S"
           "T", "U", "V", "W", "X", "Y", "Z"]

nClasses = len(classes)

xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
X_train_scale = xTrain/255.0
X_test_scale = xTest/255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(X_train_scale, yTrain)

def getPrediction(image):
    impil = Image.open(image)
    imgbw = impil.convert("L")
    imgbwresized = imgbw.resize((22, 30), Image.ANTIALIAS)

    pixelFilter = 20
    minimumPixel = np.percentile(imgbwresized, pixelFilter)
    imgbw_resized_inverted_scaled = np.clip(imgbwresized - minimumPixel, 0, 255)

    maximumPixel = np.max(imgbwresized)
    imgbw_resized_inverted_scaled = np.asarray(imgbw_resized_inverted_scaled)/maximumPixel

    testSample = np.array(imgbw_resized_inverted_scaled).reshape(1, 784)
    testPredict = clf.predict(testSample)

    return testPredict[0]