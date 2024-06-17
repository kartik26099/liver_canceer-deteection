import pandas as pd
import pandas as pdd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
df=pd.read_csv(r"C:\Users\ASUS\Downloads\Liver_disease_data.csv")
x=df.iloc[:, :-1].values
y=df.iloc[:, -1].values

classifier={"Logistic regression":LogisticRegression(max_iter=1000),
            "KNneighbhour":KNeighborsClassifier(n_neighbors=10),
            "SVM":SVC(kernel="linear"),
            "decision tree":DecisionTreeClassifier(min_samples_split=10,criterion="entropy"),
            "clasasifier tree":RandomForestClassifier(n_estimators=100)}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
model_performance = {}
cross={}
for name,models in classifier.items():
    models.fit(x_train,y_train)
    y_predict=models.predict(x_test)
    model_performance[name]=accuracy_score(y_test,y_predict)
    cross=cross_val_score(estimator=models,X=x_train,y=y_train,cv=10)
print(model_performance)
print(cross)