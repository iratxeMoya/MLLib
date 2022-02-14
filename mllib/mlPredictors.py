import itertools
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from .logUtils import logger
from .utils import *
from .models import *

class Base():
    def __init__(self, data):
        
        self.models = []
        self.data = data
        
    def visualizeModel(self, rank = 0):
        self.models[rank]['model'].visualize() 
    
    @abstractmethod 
    def featureSelection(self):
        raise NotImplementedError('Method not implemented')
    
    @abstractmethod
    def testModels(self):
        raise NotImplementedError('Method not implemented')
    
    @abstractmethod
    def createKnownModel(self):
        raise NotImplementedError('Method not implemented')


class LinearReg(Base):
    
    def __init__(self, data):
        logger.info('Initilizing Linear Regression module')
        
        super().__init__(data)
        self.featureRank = None
        
        logger.info('Linear Regression model initialized')
        
    def featureSelection(self, nf, features, predictor):
        estimator = SVR(kernel='linear')
        rfe = RFE(estimator, n_features_to_select=nf, step=1)
        rfe = rfe.fit(self.data[features], self.data[predictor])
        dictionary = sorted(list({'f': f, 'r': r, 's': s} for f, r, s in zip(features, rfe.ranking_, rfe.support_)), key=lambda k: k['r'])
        self.featureRank = [d['f'] for d in dictionary]
    
    def testModels(self, predictableColName, minGrade = 1, maxGrade = 3, notFeatures = [], maxPredictors = None):
        
        logger.info('---- TESTING MODELS ----')
        logger.info('Getting posible predictor combinations')
        
        Y = self.data[predictableColName]
        posibleFeatureColNames = self.data.loc[:, self.data.columns != predictableColName].columns.tolist()
        posibleFeatureColNames = list(filter(lambda e: filterByName(e, notFeatures), posibleFeatureColNames))
        posibleFeatureColNames = list(filter(lambda e: filterStrs(e, self.data), posibleFeatureColNames))
                
        logger.info('Testing models with degree between {} and {}'.format(minGrade, maxGrade))
        
        self.featureSelection(1, posibleFeatureColNames, predictableColName)
        maxPredictors = maxPredictors if maxPredictors else len(posibleFeatureColNames)

        for grade in range(minGrade, maxGrade + 1):
            combScores = []
            for nf in range(1, len(posibleFeatureColNames)):
                comb = self.featureRank[:nf]
                model = LinearModel(grade)
                X = self.data[comb]
                
                model.generate(X, Y)
                model.train()
                score = {'score': round(model.getScore, 2), 'error': round(model.error, 2), 'auc': model.auc, 'comb': comb, 'degree': grade, 'model': model}
                combScores.append(score)
            combScores = sorted(combScores, key=lambda k: (k['error'], -k['score'], -k['auc']))
            self.models.append(combScores[0])
        
        self.models = sorted(self.models, key=lambda k: (k['error'], -k['score'], k['degree']))
        
        logger.info('---- RESULTS ----')
        logger.info('The best found model has score {} and error {} and has been achived with predictors {} and degree {}'.format(self.models[0]['score'], self.models[0]['error'], self.models[0]['comb'], self.models[0]['degree']))
        
    def createKnownModel(self, predictableColName, numFeatures, grade):
        
        logger.info('Generating Multi-linear Regresion model to predict {} with {} features and {} grade'.format(predictableColName, numFeatures, grade))
        
        Y = self.data[predictableColName]
        posibleFeatureColNames = self.data.loc[:, self.data.columns != predictableColName].columns.tolist()
        posibleFeatureColNames = list(filter(lambda e: filterStrs(e, self.data), posibleFeatureColNames))
        
        self.featureSelection(numFeatures, posibleFeatureColNames, predictableColName)
        comb = self.featureRank[:numFeatures]
        
        model = LinearModel(grade)
        X = self.data[comb]
        
        model.generate(X, Y)
        model.train()
        
        score = {'score': round(model.getScore, 2), 'error': round(model.error, 2), 'comb': comb, 'degree': grade, 'model': model}
        self.models.append(score)
        logger.info('---- RESULTS ----')
        logger.info('The model has score {} and error {}'.format(score['score'], score['error']))
        
class LogitReg(Base):
    
    def __init__(self, data, thr):
        logger.info('Initilizing Logistic Regression module')
        
        super().__init__(data)
        self.threshold = thr
        self.featureRank = None
        
        logger.info('Logistic Regression model initialized')
        
    def featureSelection(self, nf, features, predictor):
        lm = LogisticRegression()
        rfe = RFE(lm, n_features_to_select=nf, verbose=0)
        rfe = rfe.fit(self.data[features], self.data[predictor])
        dictionary = sorted(list({'f': f, 'r': r, 's': s} for f, r, s in zip(features, rfe.ranking_, rfe.support_)), key=lambda k: k['r'])
        self.featureRank = [d['f'] for d in dictionary]
    
    def testModels(self, predictableColName, notFeatures = [], maxPredictors = None):
        logger.info('---- TESTING MODELS ----')
        logger.info('Getting posible predictors')
        
        Y = self.data[predictableColName]
        posibleFeatureColNames = self.data.loc[:, self.data.columns != predictableColName].columns.tolist()
        posibleFeatureColNames = list(filter(lambda e: filterByName(e, notFeatures), posibleFeatureColNames))
        posibleFeatureColNames = list(filter(lambda e: filterStrs(e, self.data), posibleFeatureColNames))
        
        logger.info('Testing models')
        self.featureSelection(1, posibleFeatureColNames, predictableColName)
        maxPredictors = maxPredictors if maxPredictors else len(posibleFeatureColNames)
        
        for nf in range(1, maxPredictors + 1):
            comb = self.featureRank[:nf]
            
            model = LogitModel(self.threshold)
            X = self.data[comb]
            
            model.generate(X, Y, 0.1)
            model.train()
            
            score = {'score': round(model.getScore, 2), 'error': round(model.error, 2), 'comb': comb, 'model': model}
            self.models.append(score)
        
        self.models = sorted(self.models, key=lambda k: (k['error'], -k['score']))
        
        logger.info('---- RESULTS ----')
        logger.info('The best found model has score {} and error {} and has been achived with {} predictors'.format(self.models[0]['score'], self.models[0]['error'], len(self.models[0]['comb'])))
        
    def createKnownModel(self, predictableColName, numFeatures):
        
        logger.info('Generating Logistic Regresion model to predict {} with {} features'.format(predictableColName, numFeatures))
        
        Y = self.data[predictableColName]
        posibleFeatureColNames = self.data.loc[:, self.data.columns != predictableColName].columns.tolist()
        posibleFeatureColNames = list(filter(lambda e: filterStrs(e, self.data), posibleFeatureColNames))
        
        self.featureSelection(numFeatures, posibleFeatureColNames, predictableColName)
        comb = self.featureRank[:numFeatures]
        
        model = LogitModel(self.threshold)
        X = self.data[comb]
        
        model.generate(X, Y, 0.1)
        model.train()
        
        score = {'score': round(model.getScore, 2), 'error': round(model.error, 2), 'comb': comb, 'model': model}
        self.models.append(score)
        logger.info('---- RESULTS ----')
        logger.info('The model has score {} and error {}'.format(score['score'], score['error']))
        
class Kmeans(Base):
    
    def __init__(self, data):
        logger.info('Initilizing Kmeans Classifier module')
        
        super().__init__(data)
        self.ssw = []
        
        logger.info('Kmeans Classifier model initialized')
        
    def featureSelection(self):
        return super().featureSelection()
        
    def testModels(self, minK = 1, maxK = None):
        
        logger.info('---- TESTING MODELS ----')
        maxK = maxK if maxK is not None else len(self.data)
        
        for k in range(minK, maxK + 1):
            model = KmeansModel(k)
            
            model.generate(self.data)
            model.train()
            
            score = {'model': model, 'score': round(100 - model.getScore, 2), 'k': k, 'performance': {'davies_bouldin': model.davies_bouldin, 'calinski_harabasz': model.calinski_harabasz, 'silhouette': model.silhouette}, 'isGoodK': model.isGoodK()}
            self.models.append(score)
            self.ssw.append(model.getScore)
        
        self.models.sort(key=lambda x: (not x['isGoodK'], x['performance']['davies_bouldin'], -x['performance']['calinski_harabasz'], -x['performance']['silhouette']))
        print([(m['isGoodK'], m['performance']['davies_bouldin']) for m in self.models])
        
        logger.info('---- RESULTS ----')
        logger.info('The best found model has score {} and davies_bouldin {}, calinski_harabasz {} and silhouette {} and is a {} clustering with k = {}'.format(self.models[0]['score'], self.models[0]['performance']['davies_bouldin'], self.models[0]['performance']['calinski_harabasz'], self.models[0]['performance']['silhouette'], 'GOOD' if self.models[0]['isGoodK'] else 'BAD', self.models[0]['k']))
        
    def visualizeModel(self, rank = 0):
        self.models[rank]['model'].visualize(self.ssw)
    
    def createKnownModel(self, k):
        
        logger.info('Generating KMeans Clustering model to with k = {}'.format(k))
        
        model = KmeansModel(k)
            
        model.generate(self.data)
        model.train()
        
        score = {'model': model, 'score': round(model.getScore, 2), 'k': k, 'performance': {'davies_bouldin': model.davies_bouldin, 'calinski_harabasz': model.calinski_harabasz, 'silhouette': model.silhouette}, 'isGoodK': model.isGoodK()}
        self.models.append(score)
        
        logger.info('---- RESULTS ----')
        logger.info('The model has score {} and davies_bouldin {}, calinski_harabasz {} and silhouette {} and is a {} clustering'.format(self.models[0]['score'], self.models[0]['performance']['davies_bouldin'], self.models[0]['performance']['calinski_harabasz'], self.models[0]['performance']['silhouette'], 'GOOD' if self.models[0]['isGoodK'] else 'BAD'))