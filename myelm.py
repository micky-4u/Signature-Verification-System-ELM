import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(x, w, c):
    return 1 / (1 + np.exp(-(np.dot(x, w) + c)))

def gaussian(x, w, c):
    return np.exp(-c * np.linalg.norm(x - w, 'fro'))

def hyperbolic_tangent(x, w, c):
    return (1 - np.exp(-(np.dot(w, x) + c)))/(1 + np.exp(-(np.dot(x, w) + c)))
  
#Get function
def getActivation(name):
    return {
        'sigmoid': sigmoid,
        'gaussian': gaussian,
        'hyperbolic_tangent': hyperbolic_tangent,
    }[name]

#Activate function matrix
def H(x, activate, L):
    M = x.shape[1]
    w = np.random.normal(size=(M, L))
    c = np.random.rand(L)
    act = getActivation(activate)
    return act(x, w, c)

class ELM:
    def __init__ (self,num_hidden, activation='sigmoid'):
        self.activation = getActivation(activation)
        self.L = num_hidden
        
    def fit(self,  X,  y, C=1):
        self.X = X
        self.Y = y
        self.I = np.eye(self.L, self.L) 
        self.M = X.shape[1]
        self.w =  np.random.normal(size=(self.M, self.L))
        self.c = np.random.normal(size=(self.L))
        self.C = C
        
        self.H = self.activation(self.X, self.w, self.c)
        self.Beta = np.linalg.inv(self.H.T @ self.H + self.I /self.C) @ self.H.T @ self.Y
        
    def predict(self, X):
        H_pre = self.activation(X, self.w, self.c)
        return H_pre @ self.Beta