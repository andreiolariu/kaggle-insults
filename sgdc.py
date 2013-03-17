import time
import simplejson as json

from sklearn.linear_model import SGDClassifier
import numpy as np

from nlp_dict import NLPDict

class SGDC(object):
    
    def __init__(self, texts, classes, nlpdict):
        # TODO: add list of smileys to texts/classes
        self.s = SGDClassifier(loss='hinge', penalty='l1', shuffle=True, \
                class_weight='auto')
        if nlpdict:
            self.dictionary = nlpdict
        else:
            self.dictionary = NLPDict(texts=texts)
        self._train(texts, classes)
        
    def _train(self, texts, classes):
        vectors = self.dictionary.feature_vectors(texts)
        self.s.fit(vectors, classes)
        
    def classify(self, texts):
        vectors = self.dictionary.feature_vectors(texts)
        predictions = self.s.decision_function(vectors)
        predictions = predictions / 20 + 0.5
        predictions[predictions > 1] = 1
        predictions[predictions < 0] = 0
        return predictions
        
