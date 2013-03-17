import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from nlp_dict import NLPDict
from static import *

class DictKNN(object):
    
    def __init__(self, texts, classes):
        # TODO: add list of smileys to texts/classes
        self.knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
        self.dictionary = NLPDict(texts=texts)
        
        vectors = self._build_vector(texts[0])[np.newaxis]
        for i in xrange(1, len(texts)):
            vectors = np.vstack((vectors, self._build_vector(texts[i])))
            
        self.knn.fit(vectors, classes)
        
    def _build_vector(self, text):
        item = self.dictionary.tokenize(text)
        vector = []
        words = list(stemmed_curse_words)
        words.extend(list(you_words))
        words.sort()
        for word in words:
            freq = sum([1 for word2 in item['stemmed'] if word == word2])
            vector.append(freq)
        vector.append(item['ratio'])
        vector.append(len(item['original']))
        freq_x = sum([1 for ch in item['original'] if ch == '!'])
        freq_q = sum([1 for ch in item['original'] if ch == '?'])
        freq_a = sum([1 for ch in item['original'] if ch == '*'])
        vector.append(freq_x)
        vector.append(freq_q)
        vector.append(freq_a)
        return np.array(vector)
        
    def classify(self, texts):
        vectors = self._build_vector(texts[0])[np.newaxis]
        for i in xrange(1, len(texts)):
            vectors = np.vstack((vectors, self._build_vector(texts[i])))
            
        pred_prob = self.knn.predict_proba(vectors)
        predictions = []
        for pair in pred_prob:
            predictions.append(pair[1] - pair[0])
        predictions = np.array(predictions)
        predictions = (predictions + 1) / 2
        predictions[predictions > 1] = 1
        predictions[predictions < 0] = 0     
        return predictions
        

