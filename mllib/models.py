
import numpy as np
import pandas as pd
from .logUtils import logger
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn import metrics

class Model(ABC):
    def __init__(self):
        self.lm = None
        self.error = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
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
    def generate(self):
        raise NotImplementedError('Method not implemented')
    
    @abstractmethod
    def makePredictions(self):
        raise NotImplementedError('Method not implemented')
    
    @abstractmethod
    def train(self):
        raise NotImplementedError('Method not implemented')
    
    @abstractmethod
    def visualize(self):
        raise NotImplementedError('Method not implemented')


class LinearModel(Model):
    
    def __init__(self, grade):
        super().__init__()
        
        self.grade = grade
        self.X_test_poly = None
        self.X_train_poly = None
        
    @property
    def getScore(self):
        if self.X_test_poly is not None:
            return self.lm.score(self.X_test_poly, self.Y_test)
        else:
            return self.lm.score(self.X_test, self.Y_test)
    
    def generate(self, X, Y, grade = 1, testSize = 0.2):
        self.splitTrainTest(X, Y, testSize)
        if self.grade == 1:
            self.lm = LinearRegression()
        elif self.grade > 1:
            poli = PolynomialFeatures(degree = grade)
            self.X_train_poly = poli.fit_transform(self.X_train)
            self.X_test_poly = poli.fit_transform(self.X_test)
            self.lm = LinearRegression()
            
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
            
class LogitModel(Model):
    def __init__(self, thr):
        super().__init__()
        self.threshold = thr
        self.probs = None
        self.auc = None
        
    @property
    def getScore(self):
        return self.lm.score(self.X_test, self.Y_test)
         
    def generate(self, X, Y, testSize = 0.2):
        self.splitTrainTest(X, Y, testSize)
        self.lm = LogisticRegression()
        
    def makePredictions(self):
        self.probs = self.lm.predict_proba(self.X_test)[:, 1]
        self.prediction = np.where(self.probs > self.threshold, 1, 0)
        self.auc = metrics.roc_auc_score(self.Y_test, self.probs)

    def train(self):
        self.lm.fit(self.X_train, self.Y_train)
        self.makePredictions()
        self.error = 1 - metrics.accuracy_score(self.Y_test, self.prediction)
        
    def getCorrectness(self):
        df = pd.DataFrame({'y': self.Y_test, 'pred': self.prediction})
        correct = []
        for i, row in df.iterrows():
            if row['y'] == row['pred']:
                correct.append(True)
            else:
                correct.append(False)
                
        return correct
    
    def visualize(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)
        ax.set_title("Logistic Regression Visualization")
            
        ax.set_xlabel('Samples')
        ax.set_ylabel('Predicted value')
        
        correct = self.getCorrectness()
        correctClassified = sum(correct) / len(correct)
        goodPoints = []
        badPoints = []
        for i, c in enumerate(correct):
            if c == True:
                goodPoints.append(self.probs[i])
                badPoints.append(None)
            else:
                badPoints.append(self.probs[i])
                goodPoints.append(None)
        
        ax.scatter(np.arange(len(self.X_test.values)), goodPoints, c="green", s=2, label='Correct classified: ' + str(round(correctClassified, 2)))
        ax.axhline(y = self.threshold, color='gray', linestyle = '--')
        ax.scatter(np.arange(len(self.X_test.values)), badPoints, c="red", s=2, label = 'Wrong classified: ' + str(round(1-correctClassified, 2)))
        
        ax.legend(loc="lower right")
        
        ax = fig.add_subplot(212)
        ax.set_title("ROC Curves")
        
        ax.axis(xmin=-0.01,xmax=1.01)
        ax.axis(xmin=-0.01,xmax=1.01)
            
        ax.set_xlabel("False Positive Rate")
        ax.set_xlabel("True Positive Rate")
        
        false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(self.Y_test, self.probs)

        ax.plot(false_positive_rate, true_positive_rate, 'b', label="AUC: " + str(self.auc))
        ax.fill_between(false_positive_rate, true_positive_rate, facecolor='lightblue', alpha=0.5)
        
        ax.legend(loc="lower right")
        
        plt.show()
        
        
            