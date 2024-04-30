# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output. 

## Program:
/*
Program to implement the SVM For Spam Mail Detection..

Developed by: Pavithra D

RegisterNumber:  212223230146
*/
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df= pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
df.head()

vectorizer=CountVectorizer()
X= vectorizer.fit_transform(df['v2'])
y = df['v1']

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.25, random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train, y_train)

predictions=model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
priint(classification_report(y_test, predictions))
```
## Output:

**df.head()**
![image](https://github.com/PavithraD23004871/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138955967/6bf27a3b-daac-4255-b384-4a544ad5132e)

**prediction**

![image](https://github.com/PavithraD23004871/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138955967/a58efdee-7e77-4de8-b134-695bde0c26b6)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
