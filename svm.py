import time
import simplejson as json

from sklearn import svm
import numpy as np

from nlp_dict import NLPDict

class SVM(object):
    
    def __init__(self, texts, classes, nlpdict=None):
        # TODO: add list of smileys to texts/classes
        self.svm = svm.LinearSVC(C=1000, class_weight='auto')
        if nlpdict:
            self.dictionary = nlpdict
        else:
            self.dictionary = NLPDict(texts=texts)
        self._train(texts, classes)
        
    def _train(self, texts, classes):
        vectors = self.dictionary.feature_vectors(texts)
        self.svm.fit(vectors, classes)
        
    def classify(self, texts):
        vectors = self.dictionary.feature_vectors(texts)
        predictions = self.svm.decision_function(vectors)
        predictions = np.transpose(predictions)[0]
        predictions = predictions / 2 + 0.5
        predictions[predictions > 1] = 1
        predictions[predictions < 0] = 0
        return predictions
        
