import time
import pickle
import simplejson as json

from sklearn.linear_model import LogisticRegression
import numpy as np

from nlpdictch import NLPDictCh

class ChLogReg(object):
    '''
        logistic regression
    '''
    
    def __init__(self, texts, classes, nlpdictch=None, scale=1, C=1.0):
        self.scale = scale
        self.l = LogisticRegression(penalty='l2', dual=True, C=C, \
                class_weight='auto')
        if nlpdictch:
            self.dictionary = nlpdictch
        else:
            self.dictionary = NLPDictCh(texts=texts)
        vectors = self.dictionary.feature_vectors(texts)
        self.l.fit(vectors, classes)
        
    def classify(self, texts):
        '''
            Classify a list of texts
        '''
        vectors = self.dictionary.feature_vectors(texts)
        pred_prob = self.l.predict_proba(vectors)
        predictions = []
        for pair in pred_prob:
            predictions.append(pair[1] - pair[0])
        predictions = np.array(predictions)
        predictions = (predictions + 1) / 2
        predictions *= self.scale
        predictions[predictions > 1] = 1
        predictions[predictions < 0] = 0      
        return predictions

