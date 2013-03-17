import neurolab as nl
from sklearn import cross_validation
import numpy as np

from nlp_dict import NLPDict
from static import *
                
class Ensemble(object):
    '''
        neural network using several clasifiers as input
    '''
    
    def __init__(self, texts, classes, params, models):
        self.models = models
        self.params = params
            
        self.texts = texts
        self.classes = classes
        self.dictionary = NLPDict(texts=texts)
        
        classes = np.array(classes)
        texts = np.array(texts)
        
        net_inputs = []
        expected_outputs = np.array([])
        
        print 'get training data'
        # get some training data
        for train, test in cross_validation.StratifiedKFold(classes, 5):
            texts_train = texts[train]
            classes_train = classes[train]
            # classes_train = list(classes_train)
            texts_test = texts[test]
            classes_test = classes[test]
            
            net_inputs_batch = []
            for model, params in self.models:
                m = model(texts_train, classes_train, *params)
                p = np.array(m.classify(texts_test))[np.newaxis]
                if len(net_inputs_batch):
                    net_inputs_batch = np.vstack([net_inputs_batch, p])
                else:
                    net_inputs_batch = p
            net_inputs_batch = net_inputs_batch.T
            
            vectors = self._build_vector(texts_test[0])[np.newaxis]
            for i in xrange(1, len(texts_test)):
                vectors = np.vstack((vectors, self._build_vector(texts_test[i])))
            net_inputs_batch = np.hstack((net_inputs_batch, vectors))
            
            expected_outputs = \
                    np.concatenate((expected_outputs, classes_test), axis=0)
            if len(net_inputs):
                net_inputs = \
                    np.vstack((net_inputs, net_inputs_batch))
            else:
                net_inputs = net_inputs_batch
    
        # init network
        self.net = nl.net.newff( \
            [[0, 1] for i in range(len(self.models) + 6)], \
            self.params['structure'] \
        )
        
        print 'train network'
        # train network
        expected_outputs = expected_outputs[np.newaxis].T
        nl.train.train_rprop( \
                self.net, \
                net_inputs, \
                expected_outputs, \
                epochs=self.params['epochs'] \
        )
    
    def _build_vector(self, text):
        item = self.dictionary.tokenize(text)
        vector = []
        words = list(stemmed_curse_words)
        words.extend(list(you_words))
        words.sort()
        freq = 0
        for word in words:
            freq += sum([1 for word2 in item['stemmed'] if word == word2])
        vector.append(min(freq / 10.0, 1))
        vector.append(item['ratio'])
        vector.append(min(len(item['original']) / 500.0, 1))
        freq_x = sum([1 for ch in item['original'] if ch == '!'])
        freq_q = sum([1 for ch in item['original'] if ch == '?'])
        freq_a = sum([1 for ch in item['original'] if ch == '*'])
        vector.append(min(1, freq_x / 10.0))
        vector.append(min(1, freq_q / 10.0))
        vector.append(min(1, freq_a / 10.0))
        return np.array(vector)
    
    def classify(self, texts):
        print 'classify'
        net_inputs = []
        for model, params in self.models:
            m = model(self.texts, self.classes, *params)
            p = np.array(m.classify(texts))[np.newaxis]
            if len(net_inputs):
                net_inputs = np.vstack([net_inputs, p])
            else:
                net_inputs = p
        net_inputs = net_inputs.T
                
        vectors = self._build_vector(texts[0])[np.newaxis]
        for i in xrange(1, len(texts)):
            vectors = np.vstack((vectors, self._build_vector(texts[i])))
        net_inputs = np.hstack((net_inputs, vectors))
        
        results = self.net.sim(net_inputs)
        results = results.T[0]
        return results
