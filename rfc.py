from sklearn.naive_bayes import MultinomialNB
import numpy as np

from nlp_dict import NLPDict

class RFC(object):
    
    def __init__(self, texts, classes):
        self.dictionary = NLPDict(texts=texts)
        vectors = self.dictionary.feature_vectors(texts)
        self.nb = MultinomialNB()
        self.nb.fit(vectors, classes)
        
    def classify(self, texts):
        vectors = self.dictionary.feature_vectors(texts)
        pred_prob = self.nb.predict_proba(vectors)
        predictions = []
        for pair in pred_prob:
            predictions.append(pair[1] - pair[0])
        predictions = np.array(predictions)
        predictions = (predictions + 1) / 2
        #predictions *= 0.75
        predictions[predictions > 1] = 1
        predictions[predictions < 0] = 0        
        return predictions

'''
from sklearn.naive_bayes import MultinomialNB

from nlp_dict import NLPDict
dictionary = NLPDict(texts=texts)
vectors = dictionary.feature_vectors(texts)
nb = MultinomialNB()
nb.fit(vectors, classes)
predictions = nb.predict_proba(vectors)

p2 = []
for pair in predictions:
    p2.append(pair[1] - pair[0])
p2 = np.array(p2)
p2 = (p2 + 1) / 2
'''
