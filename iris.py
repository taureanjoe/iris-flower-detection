import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.data")

le = LabelEncoder()
y = df["Iris-setosa"]
df['target'] = le.fit_transform(y)
X = df.iloc[:,:4]
y = df.iloc[:,5]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
sv = SVC(kernel='linear').fit(X_train, y_train)

pickle.dump(sv, open('iris.pkl', 'wb'))

