import itertools
from .logUtils import logger
from .utils import *
from .models import *

class LinearReg():
    
    def __init__(self, data):
        logger.info('Initilizing Linear Regression module')
        
        self.models = []
        self.bestModel = None
        self.data = data
        
        logger.info('Linear Regression model initialized')
    
    def testModels(self, predictableColName, minGrade = 1, maxGrade = 3):
        
        logger.info('---- TESTING MODELS ----')
        logger.info('Getting posible predictor combinations')
        
        Y = self.data[predictableColName]
        posibleFeatureColNames = self.data.loc[:, self.data.columns != predictableColName].columns.tolist()
        posibleFeatureColNames = list(filter(lambda e: filterStrs(e, self.data), posibleFeatureColNames))
        
        combinations = []
        for r in range(len(posibleFeatureColNames)+1):
            for combination in itertools.combinations(posibleFeatureColNames, r):
                combinations.append(list(combination))
                
        combinations = list(filter(len, combinations))
                
        logger.info('Testing models with degree between {} and {}, and all predictor combinations from {}'.format(minGrade, maxGrade, posibleFeatureColNames))
                
        for grade in range(minGrade, maxGrade + 1):
            combScores = []
            for comb in combinations:
                model = LinearModel()
                X = self.data[comb]
                
                model.generate(X, Y, grade)
                model.train()
                score = {'score': round(model.getScore, 2), 'error': round(model.error, 2), 'comb': comb, 'degree': grade, 'model': model}
                combScores.append(score)
            combScores = sorted(combScores, key=lambda k: (k['error'], -k['score']))
            self.models.append(combScores[0])
        
        self.bestModel = sorted(self.models, key=lambda k: (k['error'], -k['score'], k['degree']))[0]
        logger.info('---- RESULTS ----')
        logger.info('The best found model has score {} and error {} and has been achived with predictors {} and degree {}'.format(self.bestModel['score'], self.bestModel['error'], self.bestModel['comb'], self.bestModel['degree']))
        
    def visualizeModel(self, rank = 0):
        self.models[rank]['model'].visualize()