from PyQt5.uic import *
from PyQt5.QtWidgets import *
import pandas as pd 
import numpy as np

data = pd.read_csv('Healthcare-Diabetes.csv')

y = np.array(data.iloc[:,-1])
X = np.array(data.iloc[:,-9:-1])

y_train = y[:int(len(y)*0.8)]
y_test = y[int(len(y)*0.8):]

X_train = X[:int(len(X)*0.8)]
X_test = X[int(len(X)*0.8):]

from sklearn.preprocessing import MinMaxScaler

# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Splitting the normalized data into training and testing sets
X_train = X_normalized[:int(len(X) * 0.8)]
X_test = X_normalized[int(len(X) * 0.8):]


class LogisticRegression:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.w = None
        self.b = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def Z(self, X):
        return np.dot(X, self.w) + self.b

    def loss_function(self, X_train, y_train):
        predictions = self.sigmoid(self.Z(X_train))
        loss = -np.mean(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))
        return loss

    def gradient(self, X_train, y_train):
        m = X_train.shape[0]
        predictions = self.sigmoid(self.Z(X_train))
        error = predictions - y_train
        dw = (1 / m) * np.dot(X_train.T, error)
        db = (1 / m) * np.sum(error)
        return dw, db

    def fit(self, X_train, y_train, epochs):
        self.w = np.zeros(X_train.shape[1])  
        for i in range(epochs):
            dw, db = self.gradient(X_train, y_train)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        print(f"Loss = {self.loss_function(X_train, y_train)}")

    def predict(self, X_test, threshold=0.5):
        probabilities = self.sigmoid(self.Z(X_test))
        return (probabilities >= threshold).astype(int)
model = LogisticRegression(lr=0.1)
model.fit(X_train, y_train,10000)

def detect():
    Pregnancies = float(win.pre.text())
    Glucose = float(win.glu.text())
    BloodPressure = float(win.blo.text())
    SkinThickness = float(win.ski.text())
    Insulin = float(win.ins.text())
    BMI = float(win.bmi.text())
    DiabetesPedigreeFunction = float(win.dpf.text())
    Age = float(win.age.text())
    
    # Normalize the input data using the same scaler
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_normalized = scaler.transform(input_data)
    
    if model.predict(input_normalized):
        win.res.setText("rak 3andk el sokkker")
    else: 
        win.res.setText("salekt'ha")

     

app = QApplication([])
win = loadUi ("diabetes.ui")
win.show()
win.pred.clicked.connect (detect)
app.exec_()