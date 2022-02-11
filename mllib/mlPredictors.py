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
        self.bestModel = None
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
        self.bestModel = sorted(self.models, key=lambda k: (k['error'], -k['score'], k['degree']))[0]
        logger.info('---- RESULTS ----')
        logger.info('The best found model has score {} and error {} and has been achived with predictors {} and degree {}'.format(self.bestModel['score'], self.bestModel['error'], self.bestModel['comb'], self.bestModel['degree']))
        
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
        self.bestModel = sorted(self.models, key=lambda k: (k['error'], -k['score']))[0]
        logger.info('---- RESULTS ----')
        logger.info('The best found model has score {} and error {} and has been achived with {} predictors'.format(self.bestModel['score'], self.bestModel['error'], len(self.bestModel['comb'])))
        
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
    