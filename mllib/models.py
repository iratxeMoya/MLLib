
import numpy as np
import pandas as pd
from .logUtils import logger
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA

class Model(ABC):
    def __init__(self):
        self.lm = None
        self.error = None
        self.X_train = None
        self.X_train_poly = None
        self.Y_train = None
        self.X_test = None
        self.X_test_poly = None
        self.Y_test = None
        self.prediction = None
        
    def splitTrainTest(self, X, Y, testSize):
        r = np.random.randn(len(X))
        check = (r>=testSize)
        
        self.X_train = X[check]
        self.X_test = X[~check]
        self.Y_train = Y[check]
        self.Y_test = Y[~check]
    
    @abstractmethod  
    def generate(self, X, Y, grade = 1, testSize = 0.2):
        raise NotImplementedError('Method not implemented')
    
    @abstractmethod
    def train(self):
        raise NotImplementedError('Method not implemented')
    
    @abstractmethod
    def visualize(self):
        raise NotImplementedError('Method not implemented')


class LinearModel(Model):
    
    def __init__(self):
        super().__init__()
        self.grade =  1
        
    @property
    def getScore(self):
        if self.X_test_poly is not None:
            return self.lm.score(self.X_test_poly, self.Y_test)
        else:
            return self.lm.score(self.X_test, self.Y_test)
    
    def generate(self, X, Y, grade = 1, testSize = 0.2):
        self.splitTrainTest(X, Y, testSize)
        if grade == 1:
            self.lm = LinearRegression()
        elif grade > 1:
            poli = PolynomialFeatures(degree = grade)
            self.X_train_poly = poli.fit_transform(self.X_train)
            self.X_test_poly = poli.fit_transform(self.X_test)
            self.lm = LinearRegression()
            self.grade = grade
            
    def makePredictions(self):
        if self.X_test_poly is not None:
            self.prediction = self.lm.predict(self.X_test_poly)
        else:
            self.prediction = self.lm.predict(self.X_test)
            
    def train(self):
        if self.X_train_poly is not None:
            self.lm.fit(self.X_train_poly, self.Y_train)
        else:
            self.lm.fit(self.X_train, self.Y_train)
        self.makePredictions()
        SSD = np.sum((self.Y_test - self.prediction) ** 2)
        RSE = np.sqrt(SSD/(len(self.X_test) - self.grade))
        self.error = RSE / np.mean(self.Y_test)
        
    def visualize(self):
        
        if len(self.X_train.columns.tolist()) == 2:
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Multi-linear Regression Visualization")
            
            ax.set_xlabel(self.X_test.columns.tolist()[0])
            ax.set_ylabel(self.X_test.columns.tolist()[1])
            ax.set_zlabel(self.Y_test.name)
            
            ax.scatter(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], self.Y_test, c="red")
            ax.plot(self.X_test.iloc[:, 0], self.X_test.iloc[:, 1], self.prediction, color='blue')
            
            plt.show()
            
        elif len(self.X_train.columns.tolist()) == 1:
            
            plt.plot(self.X_test, self.Y_test, 'ro')
            plt.plot(self.X_test, self.prediction, color='blue')
            
            plt.show()
            
        else:
            
            x = self.X_test.values
            y = self.X_test.values
            
            x = StandardScaler().fit_transform(x)
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(x)
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("Multi-linear Regression Visualization")
            
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_zlabel(self.Y_test.name)
            
            ax.scatter(principalComponents[:, 0], principalComponents[:, 1], self.Y_test, c="red")
            ax.plot(principalComponents[:, 0], principalComponents[:, 1], self.prediction, color='blue')
            
            plt.show()
            