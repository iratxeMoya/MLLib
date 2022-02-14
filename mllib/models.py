from random import sample
from .logUtils import logger
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pytz import NonExistentTimeError
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
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
        if Y is not None:
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
        
class KmeansModel(Model):
    
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.X = None
        self.davies_bouldin = 0
        self.calinski_harabasz = 0
        self.silhouette = 0
        self.centers = None
        self.pred = None
        
    @property
    def getScore(self):
        return sum(np.min(cdist(self.X_test, self.lm.cluster_centers_, "euclidean"), axis = 1))
    
    def makePredictions(self):
        self.pred = self.lm.predict(self.X_test)
    
    def getPerformance(self):
        if self.k > 1:
            self.davies_bouldin = davies_bouldin_score(self.X_test, self.pred)
            self.calinski_harabasz = calinski_harabasz_score(self.X_test, self.pred)
            self.silhouette = silhouette_score(self.X_test, self.pred)
        
    def generate(self, X, Y, testSize = 0.2):
        self.splitTrainTest(X, Y, testSize)
        self.lm = KMeans(n_clusters = self.k)
        
    def isGoodK(self):
        if self.k > 1:
            sample_silhouette_values = silhouette_samples(self.X_test, self.pred)
            ssv = []
            
            for i in range(self.k):
                ssv.append(sample_silhouette_values[self.pred == i])
            
            for v in ssv:
                if 1 - abs(self.silhouette - np.mean(v)) < 0.9:
                    return False
                
            return True
        else:
            return False
        
    def train(self):
        if len(self.X_train) >= self.k:
            self.lm.fit(self.X_train)
            
            self.centers = pd.DataFrame(self.lm.cluster_centers_)
            
            self.makePredictions()
            self.getPerformance()
        else:
            logger.error('Number of samples {} cannot be smaller than number of clusters {}'.format(len(self.X_train), self.k))
        
    def visualize(self, ssw = None):
        cmap = cm.get_cmap("Spectral")
        color_palette = [cmap(float(i)/self.k) for i in range(1, self.k + 1)]
        test_label_color = [color_palette[i] for i in self.pred]
        train_label_color = [color_palette[i] for i in self.lm.labels_]
        
        if self.X_test.shape[1] == 2:
            if ssw:
                
                fig, (ax1, ax2) = plt.subplots(1,2)
                fig.set_size_inches(20,8)
                
                ax1.set_title("Clustering for k = {}".format(self.k))
                ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], c = train_label_color)
                ax1.scatter(self.X_test[:, 0], self.X_test[:, 1], c = test_label_color, marker = '*')
                ax1.scatter(self.centers[0], self.centers[1], marker = "x")
                
                ax2.plot(np.arange(len(ssw)), ssw, "bx-")
                ax2.set_xlabel("k")
                ax2.set_ylabel("SSw(k)")
                ax2.set_title("Elbow method")
            else:
                fig, (ax1) = plt.subplots(1,1)
                fig.set_size_inches(20,8)
                
                ax1.set_title("Clustering for k = {}".format(self.k))
                ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], c = train_label_color)
                ax1.scatter(self.X_test[:, 0], self.X_test[:, 1], c = test_label_color, marker = '*')
                ax1.scatter(self.centers[0], self.centers[1], marker = "x")
            
            plt.show()
        
        elif self.X_test.shape[1] == 3:
            if ssw:
                fig = plt.figure()
                fig.set_size_inches(20,8)
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.set_title("Clustering for k = {}".format(self.k))
                ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], self.X_train[:, 2], c = train_label_color)
                ax1.scatter(self.X_test[:, 0], self.X_test[:, 1], self.X_test[:, 2], c = test_label_color, marker = '*')
                ax1.scatter(self.centers[0], self.centers[1], self.centers[2], marker = "x")
                
                ax2 = fig.add_subplot(122)
                ax2.plot(np.arange(len(ssw)), ssw, "bx-")
                ax2.set_xlabel("k")
                ax2.set_ylabel("SSw(k)")
                ax2.set_title("Elbow method")
               
            else:
                fig = plt.figure()
                fig.set_size_inches(20,8)
                ax1 = fig.add_subplot(111, projection='3d')
                ax1.set_title("Clustering for k = {}".format(self.k))
                ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], self.X_train[:, 2], c = train_label_color)
                ax1.scatter(self.X_test[:, 0], self.X_test[:, 1], self.X_test[:, 2], c = test_label_color, marker = '*')
                ax1.scatter(self.centers[0], self.centers[1], self.centers[2], marker = "x")
                
            plt.show()
        
        else:
            logger.error('Cannot visualize data with {} dimensions'.format(self.X.shape()[1]))
            
        
class AffPropModel(Model):
    
    def __init__(self):
        super().__init__()
        self.n_clust = None
        self.homogeneity = 0
        self.silhouette = 0
        self.completeness = 0
        self.vmeasure = 0
        self.ar2 = 0
        self.ami = 0
        self.centers = None
        self.centers_idx = None
        self.pred = None
        
    @property
    def getScore(self):
        return np.mean([self.homogeneity, self.completeness, self.vmeasure, self.ar2, self.ami, self.silhouette]) if self.n_clust > 1 else 0
    
    def makePredictions(self):
        self.pred = self.lm.predict(self.X_test)
    
    def getPerformance(self, makePrint = False, saveInfo = True):
        
        if self.n_clust > 1 and self.n_clust < len(self.X_test):
            if saveInfo:
                self.homogeneity = metrics.homogeneity_score(self.Y_test, self.pred)
                self.completeness = metrics.completeness_score(self.Y_test, self.pred)
                self.vmeasure = metrics.v_measure_score(self.Y_test, self.pred)
                self.ar2 = metrics.adjusted_rand_score(self.Y_test, self.pred)
                self.ami = metrics.adjusted_mutual_info_score(self.Y_test, self.pred)
                self.silhouette = metrics.silhouette_score(self.X_test, self.pred, metric="sqeuclidean")
            
            if makePrint:
                if self.n_clust:
                    logger.info("Cluster nums: %d" %self.n_clust)
                    logger.info("Homogeneity: %0.3f" %self.homogeneity)
                    logger.info("Completeness: %0.3f"%self.completeness)
                    logger.info("V-measure: %0.3f"%self.vmeasure)
                    logger.info("Adjusted R2: %0.3f"%self.ar2)
                    logger.info("Adjusted mutual information: %0.3f"%self.ami)
                    logger.info("Silhouett score: %0.3f"%self.silhouette)
                else:
                    logger.warning('Info not saved yet') 
        else:
            logger.warning('Performance cannot be computed with {} clusters'.format(self.n_clust))   
        
    def generate(self, X, Y, pref, testSize = 0.2):
        self.splitTrainTest(X, Y, testSize)
        self.lm = AffinityPropagation(preference=pref * 10)
            
    def train(self):
        self.lm.fit(self.X_train)
        
        self.centers_idx = self.lm.cluster_centers_indices_
        self.centers = pd.DataFrame(self.lm.cluster_centers_)
        self.n_clust = len(self.centers_idx)
        
        self.makePredictions()
        self.getPerformance()
        
    def visualize(self, ssw = None):
        cmap = cm.get_cmap("Spectral")
        color_palette = [cmap(float(i)/self.n_clust) for i in range(1, self.n_clust + 1)]
        
        if len(color_palette) == 0:
            color_palette.append((0.993464, 0.747712, 0.435294))
            
        test_label_color = [color_palette[i] for i in self.pred]
        train_label_color = [color_palette[i] for i in self.Y_train]
        
        if self.X_test.shape[1] == 2:
            
            plt.figure(figsize=(16,9))
            
            plt.title("Clustering for k = {}".format(self.n_clust))
            plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c = train_label_color)
            plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c = test_label_color, marker = '*')
            plt.scatter(self.centers[0], self.centers[1], marker = "x")
            
            plt.title("Clustering for %d clusters"%self.n_clust)
            
            plt.show()
        
        elif self.X_test.shape[1] == 3:
            
            plt.figure(figsize=(16,9))
            plt.clf()
            
            ax1 = fig.add_subplot(111, projection='3d')
            
            ax1.set_title("Clustering for k = {}".format(self.n_clust))
            ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], self.X_train[:, 2], c = train_label_color)
            ax1.scatter(self.X_test[:, 0], self.X_test[:, 1], self.X_test[:, 2], c = test_label_color, marker = '*')
            ax1.scatter(self.centers[0], self.centers[1], self.centers[2], marker = "x")
            
            ax1.set_title("Clustering for %d clusters"%self.n_clust)
            
            plt.show()

        
        else:
            logger.error('Cannot visualize data with {} dimensions'.format(self.X.shape()[1]))
            